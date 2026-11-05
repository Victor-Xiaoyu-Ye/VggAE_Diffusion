"""Learned transport data: heldout groups and future-target isolation."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from utils.edit_transport_data import (build_split, prepare_inputs,
    prepare_training_pair, prepare_stage63_case)


def bank_fixture():
    rows = [dict(id=f'train{i:02d}', split='train', group=f'g{i//2}', file=f'native/t{i}.pt')
            for i in range(32)]
    rows.extend([dict(id='dev', split='dev', group='devgroup', file='native/dev.pt'),
                 dict(id='benchmark', split='benchmark_dev', group='benchmark')])
    return dict(identity='native-bank', records=rows,
        manifest=dict(records=rows+[dict(id='reserved', split='test', group='testgroup')]))


def fake_encode(rae, encoder, frames):
    b,t,c,h,w = frames.shape
    result = F.adaptive_avg_pool2d(frames.reshape(b*t,c,h,w),(2,2))
    return result.permute(0,2,3,1).reshape(b,t,2,2,c)


def fake_flow(source, anchor, support, cfg):
    return dict(proposal=(source+(anchor-source[0])*support).clamp(0,1),
        confidence=torch.ones(len(source),1,*source.shape[-2:]),
        masks=support[None].expand(len(source),-1,-1,-1),
        metadata=dict(correspondence_source='source_rgb_only',fixture=True))


def sample_fixture():
    source=torch.full((5,3,16,32),40,dtype=torch.uint8)
    source[:,0]+=torch.arange(5,dtype=torch.uint8)[:,None,None]
    target=source.clone();target[:,2,:,8:24]+=80
    spatial=dict(canvas=[16,32],valid_rect=[0,0,16,32])
    return dict(source=source,target=target,timing=dict(spatial=spatial),
        row=dict(id='train0',split='train'))


class EditTransportDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return dict(seed=42,frames=5,rae_frames=3,train_pairs=0,cases=24,
            anchor_support_threshold=.06,support_dilation=1,
            training=dict(monitor_fraction=.25),transport={})

    def test_split_is_deterministic_and_whole_group_disjoint(self):
        bank=bank_fixture();cfg=self.config()
        a=build_split(bank,cfg)
        permuted=copy.deepcopy(bank);permuted['records'].reverse()
        b=build_split(permuted,cfg)
        self.assertEqual(a,b)
        self.assertEqual(a['counts'],dict(train=24,monitor=8,dev=1))
        groups={p:{r['group'] for r in a['records'] if r['partition']==p}
                for p in ('train','monitor','dev')}
        self.assertFalse(groups['train']&groups['monitor'])
        self.assertFalse((groups['train']|groups['monitor'])&groups['dev'])
        self.assertNotIn('reserved',{r['id'] for r in a['records']})
        self.assertNotIn('benchmark',{r['id'] for r in a['records']})
        cfg['train_pairs']=5
        c=build_split(bank,cfg)
        self.assertEqual(c['counts']['train']+c['counts']['monitor'],6)
        self.assertEqual(c['counts']['dev'],1)

    def test_original_reserved_groups_checked_without_cached_test(self):
        bank=bank_fixture();bank['manifest']['records'][-1]['group']='g0'
        with self.assertRaisesRegex(ValueError,'overlap'):build_split(bank,self.config())

    def test_no_training_partition_can_use_dev_group(self):
        bank=bank_fixture();bank['records'][-2]['group']='g1'
        with self.assertRaises(ValueError):build_split(bank,self.config())

    def test_single_group_keeps_training_and_zero_monitor(self):
        bank=bank_fixture();bank['records']=bank['records'][:2]+bank['records'][-2:]
        split=build_split(bank,self.config())
        self.assertEqual(split['counts'],dict(train=2,monitor=0,dev=1))

    @patch('utils.editing_video_transport.propagate_reference',side_effect=fake_flow)
    @patch('audit_edit_transport.encode_window',side_effect=fake_encode)
    def test_future_targets_change_only_supervision(self,encode,flow):
        cfg=self.config();first=sample_fixture()
        a,sa,meta=prepare_training_pair(first,cfg,None,None)
        second=copy.deepcopy(first);second['target'][1:]=220
        b,sb,_=prepare_training_pair(second,cfg,None,None)
        for key in a:
            if isinstance(a[key],torch.Tensor):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            else:self.assertEqual(a[key],b[key])
        self.assertFalse(torch.equal(sa['target'],sb['target']))
        self.assertEqual(a['observed_indices'],[0,2,4])
        self.assertEqual(a['source'].dtype,torch.uint8)
        self.assertEqual(a['source'].shape,(3,3,518,518))
        self.assertEqual(a['source_z'].shape,(3,2,2,3))
        self.assertEqual(a['anchor_z'].shape,(1,2,2,3))
        self.assertEqual(a['valid_rect'],[129,0,259,518])
        self.assertNotIn('target',a)
        self.assertFalse(meta['future_target_conditioned'])
        self.assertEqual([c.args[2].shape[1] for c in encode.call_args_list],[3,1,3,1])
        self.assertFalse(a['source_z'].requires_grad)

    @patch('utils.editing_video_transport.propagate_reference',side_effect=fake_flow)
    @patch('audit_edit_transport.encode_window',side_effect=fake_encode)
    def test_dev_training_reference_is_rejected(self,encode,flow):
        sample=sample_fixture();sample['row']['split']='dev'
        with self.assertRaisesRegex(ValueError,'dev/test'):
            prepare_training_pair(sample,self.config(),None,None)
        sample['row']['split']='train'
        with self.assertRaisesRegex(ValueError,'training/monitor'):
            prepare_training_pair(sample,self.config(),None,None,partition='dev')
        value,_,_=prepare_training_pair(sample,self.config(),None,None,partition='monitor')
        self.assertIn('heldout_train_groups',value['protocol'])

    @patch('utils.editing_video_transport.propagate_reference',side_effect=fake_flow)
    @patch('audit_edit_transport.encode_window',side_effect=fake_encode)
    def test_reference_encoding_uses_only_one_frame_and_exact_first_effect(self,encode,flow):
        sample=sample_fixture();source=sample['source'];anchor=sample['target'][0]
        support=torch.ones(1,16,32)*.35
        inputs=prepare_inputs(source,anchor,support,sample['timing']['spatial'],self.config(),None,None)
        torch.testing.assert_close(inputs['proposal'][0],inputs['anchor'],rtol=0,atol=0)
        self.assertEqual(encode.call_args_list[1].args[2].shape,(1,1,3,518,518))
        self.assertNotIn('target',inputs)
        self.assertNotIn('flows_backward',inputs)

    @patch('utils.editing_video_transport.propagate_reference',side_effect=fake_flow)
    @patch('audit_edit_transport.encode_window',side_effect=fake_encode)
    def test_stage63_reference_loader_is_read_only_and_target_free(self,encode,flow):
        sample=sample_fixture()
        class Poison(dict):
            def __getitem__(self,key):
                if key=='target':raise AssertionError('Target read by development input preparation')
                return super().__getitem__(key)
        payload=Poison(source=sample['source'],anchor=sample['target'][0],
            support=torch.ones(1,16,32),spatial=sample['timing']['spatial'],provenance='model_output')
        class Store:
            def json(self,name,required):
                self.name=name
                return dict(identity='stage63-identity',cases=[dict(id='dev')])
            def load(self,name,identity,required):
                self.loaded=(name,identity,required)
                return payload
            def publish(self,*_):raise AssertionError('Mutated historical Stage63 store')
        store=Store()
        value=prepare_stage63_case(store,'dev',self.config(),None,None)
        self.assertEqual(store.loaded,('anchors/dev/result.pt','stage63-identity',True))
        self.assertEqual(value['reference_provenance'],'model_output')
        self.assertEqual(value['case_id'],'dev')
        with self.assertRaisesRegex(ValueError,'absent'):
            prepare_stage63_case(store,'missing',self.config(),None,None)


if __name__=='__main__':unittest.main()
