"""Stage65 integration with real tiny native VACE/UniPC and explicit mock codecs.

The mock RGB codecs replace unavailable production weights only. Source optical
flow, RAE state calibration/alignment, VACE forward/backward/sampling, loss,
checkpointing, media writing, and dual ArtifactStore are the production paths.
No accelerator performance or production-quality claim is made by these tests.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_vace_edit_state as runner
from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_state_runtime import backbone
from tests.test_vace_state_data import TinyRAE
from utils.editing_data import rae_canvas, file_hash
from utils.editing_video_transport import propagate_reference
from utils.native_audit_io import sha256
from utils.scene_run import ArtifactStore
from utils.vace_state_runtime import StateConditionedVACE

REPO=Path(__file__).resolve().parents[1]
SCRATCH=REPO.parent/'VggAE_DataAudit'/'stage65_validation'


class FirstFrameOnly:
    """Saveable paired training field whose future frames must never be read."""
    def __init__(self,first):self.first=first
    def __getitem__(self,index):
        if type(index) is not int or index!=0:
            raise AssertionError('Future target entered a conditioning path')
        return self.first


class TargetPoison(dict):
    def __getitem__(self,key):
        if key in ('target','future_target','clean'):
            raise AssertionError('Dev future target entered inference')
        return super().__getitem__(key)
    def get(self,key,default=None):
        if key in ('target','future_target','clean'):
            raise AssertionError('Dev future target entered inference')
        return super().get(key,default)


class TinyCodec:
    """Explicit causal-stride4/spatial-stride8 fixture, independently coded."""
    def encode(self,values):
        result=[]
        for v in values:
            sampled=v[:,::4].permute(1,0,2,3)
            pooled=F.avg_pool2d(sampled,8).permute(1,0,2,3)
            result.append(pooled.repeat(6,1,1,1)[:16]+.125)
        return result
    def decode(self,values):
        return [F.interpolate((v[:3]-.125)[None].float(),
            size=(4*(v.shape[1]-1)+1,v.shape[2]*8,v.shape[3]*8),
            mode='trilinear',align_corners=False)[0] for v in values]


def encode_rae(rae,encoder,frames):
    b,t,c,h,w=frames.shape
    z=F.adaptive_avg_pool2d(frames.flatten(0,1),(4,4)).permute(0,2,3,1)
    return z.repeat(1,1,1,86)[...,:256].reshape(b,t,4,4,256)


def fixed_backbone(*_):
    with torch.random.fork_rng():
        torch.manual_seed(193)
        return backbone()


def fixture(root,broken=False,future_shift=0.):
    """Reusable Stage60/63/64 filesystem inputs; no production assets needed."""
    cfg=json.loads((REPO/'configs/vace_edit_state_v1.json').read_text())
    cfg.update(cases=2 if broken else 1,sample_steps=3,lora_rank=2,lora_alpha=2,
        bridge_width=8,support_dilation=1)
    cfg['transport'].update(flow_backend='farneback',flow_direct_anchor=False)
    cfg['calibration'].update(steps=2)
    cfg['training'].update(max_steps=2,min_steps=2,lr=.003,warmup_steps=1,
        accum=1,save_every=1,eval_every=1,preview_every=0,log_every=1,
        monitor_cases=1,monitor_sigmas=[.4],text_dropout=0.,state_dropout=0.,
        identity_fraction=0.,patience=5)
    data=ArtifactStore(root/'bank',[]);transport=ArtifactStore(root/'transport',[])
    refs=ArtifactStore(root/'references',[])
    for name in ('rae.pt','encoder.pt'):(root/name).write_bytes(('fixture-'+name).encode())
    args=SimpleNamespace(data=str(data.local),data_root=[],transport=str(transport.local),
        transport_root=[],references=str(refs.local),reference_root=[],
        rae=str(root/'rae.pt'),encoder=str(root/'encoder.pt'),model='mock-native-asset',
        arm='native',stop_after=0)
    bank=dict(identity='bank-v1',model_signature={'fixture':'native'},records=[])
    refs.write_json('manifest.json',dict(identity='reference-v1'))
    old=dict(identity='transport-v1',bank=bank['identity'],references='reference-v1',
        teacher=dict(rae=file_hash(args.rae),encoder=file_hash(args.encoder)),
        config=dict(support_dilation=1,anchor_support_threshold=.03),split=dict(records=[]))
    for index,part in enumerate(('train','monitor','dev')):
        key=part+'-0';row=dict(id=key,partition=part,group=part+'-group',
            file='samples/'+key+'.pt',instruction='Make the center blue')
        source=torch.full((17,3,16,32),.25+index*.03)
        source[:,0]+=torch.arange(17)[:,None,None]*.003
        source[:,1]+=torch.linspace(0,.1,32)[None,None,:]
        source=(source*255).round().byte().float()/255
        anchor=source[0].clone();anchor[2,3:13,8:24]+=.35
        support=(anchor-source[0]).abs().amax(0,keepdim=True).gt(0).float()
        # Match Stage64's existing supported first-frame reference contract.
        from edit_reference_video import source_support
        spatial=dict(canvas=[16,32],valid_rect=[0,0,16,32])
        support,_=source_support(source,anchor,spatial,None,old['config'])
        anchor=source[0]*(1-support)+anchor*support
        square,rect=rae_canvas(source[::2],spatial['valid_rect'],size=32)
        ar,_=rae_canvas(anchor[None],spatial['valid_rect'],size=32)
        sup,_=rae_canvas(support[None].expand(1,3,-1,-1),spatial['valid_rect'],size=32)
        propagated=propagate_reference(square,ar[0],sup[:,:1][0],cfg['transport'])
        inputs=dict(source=(square*255).round().byte(),
            proposal=(propagated['proposal']*255).round().byte(),
            source_z=encode_rae(None,None,square[None])[0],
            anchor_z=encode_rae(None,None,ar[None])[0],anchor=ar[0],
            masks=propagated['masks'],confidence=propagated['confidence'],
            support=sup[0,:1],observed_indices=list(range(0,17,2)),
            valid_rect=rect,source_spatial=spatial)
        text=torch.linspace(-.3,.5,48).reshape(4,12)
        negative=torch.zeros(3,12)
        target=source.clone();target[:,2,3:13,8:24]+=.35
        target[1:]=(target[1:]+future_shift).clamp(0,1)
        target[0]=anchor
        if part!='dev':
            item=dict(source=(source*255).round().byte(),
                target=FirstFrameOnly((anchor*255).round().byte()),text=text,negative=negative,
                clean=TinyCodec().encode([(target*2-1).permute(1,0,2,3)])[0],
                timing=dict(spatial=spatial))
            data.save(row['file'],item,bank['identity'])
        else:
            refs.save('anchors/'+key+'/result.pt',TargetPoison(source=source,anchor=anchor,
                text=text,negative=negative,provenance='model_output'),refs.json('manifest.json')['identity'])
        transport.save('inputs/'+key+'.pt',inputs,old['identity'])
        old['split']['records'].append(row);bank['records'].append(row)
    if broken:
        old['split']['records'].insert(0,dict(id='dev-bad',partition='dev',group='bad-group',file='absent.pt'))
    transport.write_json('manifest.json',old);data.write_json('bank.json',bank)
    out=ArtifactStore(root/'local',[str(root/'primary'),str(root/'mirror')])
    return cfg,args,out


@contextlib.contextmanager
def mock_assets():
    """Only heavyweight asset IO/codecs are mocked; all Stage65 logic is real."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(cpu_native_import())
        stack.enter_context(patch.object(runner,'validate_assets',return_value={'fixture':'native'}))
        stack.enter_context(patch.object(runner,'load_backbone',side_effect=fixed_backbone))
        stack.enter_context(patch.object(runner,'load_vae',side_effect=lambda *_:TinyCodec()))
        stack.enter_context(patch.object(runner,'load_teacher',side_effect=lambda *_:TinyRAE()))
        stack.enter_context(patch.object(runner,'load_encoder',return_value=nn.Identity()))
        stack.enter_context(patch('audit_edit_transport.encode_window',side_effect=encode_rae))
        yield


class StatePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);SCRATCH.mkdir(parents=True,exist_ok=True)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(mock_assets())
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.device=torch.device('cpu')

    def prepare(self,root=None,**kw):
        root=root or self.root;root.mkdir(parents=True,exist_ok=True)
        cfg,args,out=fixture(root,**kw)
        runner.states(args,cfg,out,self.device,0,1)
        self.assertTrue(all(r['status']=='completed' for r in out.json('phases/states.json')['records']))
        runner.cache(args,cfg,out,self.device,0,1)
        self.assertTrue(all(r['status']=='completed' for r in out.json('phases/cache.json')['records']))
        return cfg,args,out

    def test_end_to_end_train_render_lossless_and_dual_repair(self):
        cfg,args,out=self.prepare();m=runner.checked_manifest(out,cfg)
        records=out.json('phases/cache.json')['records'];row=next(r for r in records if r['partition']=='train')
        prepared=out.load(row['state_file'],m['identity'],True)
        self.assertEqual(prepared['calibration']['fallback_count'],0)
        self.assertLess(prepared['calibration']['after'],prepared['calibration']['before'])
        rae=runner.load_case(out,row,m['identity'],'rae',self.device,supervised=True)
        rgb=runner.load_case(out,row,m['identity'],'rgb',self.device,supervised=True)
        self.assertEqual(tuple(rae['state'].shape),(514,6,1,2))
        self.assertEqual(tuple(rgb['state'].shape),(1538,6,1,2))
        self.assertEqual(tuple(rae['clean'].shape),(16,6,2,4))
        torch.testing.assert_close(rae['context'],rgb['context'],rtol=0,atol=0)
        torch.testing.assert_close(rae['clean'][:,:1],rae['context'][:16,:1].float(),check_dtype=False)
        for arm in ('rgb','rae'):
            args.arm=arm
            torch.manual_seed(cfg['seed'])
            model=StateConditionedVACE(fixed_backbone(),cfg,cfg['state_channels'][arm])
            before=model.deltas();frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
            runner.train_loop(args,cfg,out,m,records,model,self.device,0,1)
            trained=model.deltas()
            self.assertGreater(float(trained['bridge.2.weight'].abs().sum()),0)
            self.assertTrue(any(not torch.equal(before[n],trained[n]) for n in trained if n.endswith('.b')))
            for n,p in model.named_parameters():
                if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
            self.assertEqual(runner.arm_store(out,arm).json('training_summary.json')['updates'],2)
            chosen=runner.arm_store(out,arm).load('selected.pt',runner.digest([m['identity'],arm]),True)
            model.restore(chosen['model'])
            runner.render_records(args,cfg,out,m,records,self.device,0,1,model=model)
        args.arm='native';runner.render(args,cfg,out,self.device,0,1)
        try:runner.summarize(out,cfg)
        except RuntimeError as exc:raise AssertionError(json.dumps(out.json('summary.json'),indent=2)) from exc
        self.assertEqual(out.json('summary.json')['status'],'completed')
        identities=[]
        for arm in ('native','rgb','rae'):
            folder='render/'+arm+'/dev-0'
            record=out.json('phases/render_'+arm+'.json')['cases'][0]
            self.assertEqual(record['status'],'completed');identities.append(record['input_identity'])
            payload=out.load(folder+'/result.pt',m['identity'],True)
            self.assertEqual(tuple(payload['generated'].shape),(17,3,16,32))
            archive=np.load(out.local/folder/'generated_frames.npz')
            self.assertEqual(int(archive['fps']),8)
            np.testing.assert_array_equal(archive['frames'],runner.crop_video(payload['generated'],[0,0,16,32]).numpy())
            for name in (folder+'/comparison.mp4',folder+'/generated_frames.npz'):
                self.assertEqual(sha256(out.local/name),sha256(Path(out.roots[0])/name))
                self.assertEqual(sha256(out.local/name),sha256(Path(out.roots[1])/name))
        self.assertEqual(len(set(identities)),1)
        # Interrupt exactly between the already-committed MP4/metrics receipt
        # and lossless NPZ publication: resume must use saved generated RGB.
        folder='render/native/dev-0';receipt_name=folder+'/receipt.json'
        receipt=out.json(receipt_name);lossless=folder+'/generated_frames.npz'
        files=[name for name in receipt['files'] if name!=lossless]
        runner.seal(out.local,receipt_name,m['identity'],files,receipt['metadata'])
        out.publish([receipt_name])
        for root in [out.local,*out.roots]:(Path(root)/lossless).unlink()
        args.arm='native'
        with patch.object(runner,'load_backbone',side_effect=AssertionError('NPZ repair must not resample')):
            runner.render(args,cfg,out,self.device,0,1)
        self.assertIn(lossless,out.json(receipt_name)['files'])
        self.assertEqual(len({sha256(Path(root)/lossless) for root in [out.local,*out.roots]}),1)
        # Intermediate preview receipts receive the same summary-time repair
        # as final media, including independent missing local/durable copies.
        args.arm='rae';prefix='previews/rae/s000002'
        runner.render_records(args,cfg,out,m,records,self.device,0,1,model=model,
            checkpoint_tag='ema_step_2',prefix=prefix)
        preview_movie=prefix+'/dev-0/comparison.mp4'
        preview_npz=prefix+'/dev-0/generated_frames.npz'
        (out.local/preview_movie).unlink();(Path(out.roots[0])/preview_movie).unlink()
        (Path(out.roots[1])/preview_npz).unlink()
        runner.summarize(out,cfg)
        self.assertIn(prefix,out.json('previews/index.json')['prefixes'])
        self.assertEqual(out.json('summary.json')['preview_failures'],[])
        for name in (preview_movie,preview_npz):
            self.assertEqual(len({sha256(Path(root)/name) for root in [out.local,*out.roots]}),1)
        movie='render/rae/dev-0/comparison.mp4';lossless='render/rae/dev-0/generated_frames.npz'
        (out.local/movie).unlink();(Path(out.roots[0])/movie).unlink();(Path(out.roots[1])/lossless).unlink()
        args.arm='rae'
        with patch.object(runner,'load_backbone',side_effect=AssertionError('A complete render must be recovered')):
            runner.render(args,cfg,out,self.device,0,1)
        runner.summarize(out,cfg)
        for name in (movie,lossless):
            hashes=[sha256(Path(root)/name) for root in [out.local,*out.roots]]
            self.assertEqual(len(set(hashes)),1)
        # A selected-model change invalidates generated output even if all old
        # files/receipts remain. This test uses a real new model-state receipt.
        store=runner.arm_store(out,'rae');identity=runner.digest([m['identity'],'rae'])
        chosen=store.load('selected.pt',identity,True);chosen['model']['bridge.2.bias']+=.01
        store.save('selected.pt',chosen,identity)
        with patch.object(runner,'load_backbone',wraps=fixed_backbone) as load:
            runner.render(args,cfg,out,self.device,0,1)
        self.assertEqual(load.call_count,1)
        runner.summarize(out,cfg)

    def test_future_target_changes_labels_only_and_config_invalidates(self):
        cfg,a,out=self.prepare(self.root/'original')
        other,b,second=self.prepare(self.root/'changed_future',future_shift=.3)
        m=out.json('manifest.json');n=second.json('manifest.json')
        self.assertEqual(m['identity'],n['identity'])
        for row in out.json('phases/cache.json')['records']:
            first=out.load(row['state_file'],m['identity'],True)
            changed=second.load(row['state_file'],n['identity'],True)
            torch.testing.assert_close(first['rae_state'],changed['rae_state'],rtol=0,atol=0)
            for key in ('source','reference','anchor','masks','confidence'):
                torch.testing.assert_close(first['reference'][key],changed['reference'][key],rtol=0,atol=0)
            first=out.load(row['cache_file'],m['identity'],True);changed=second.load(row['cache_file'],n['identity'],True)
            torch.testing.assert_close(first['context'],changed['context'],rtol=0,atol=0)
            if row['partition']!='dev':
                first=out.load(row['supervision_file'],m['identity'],True)
                changed=second.load(row['supervision_file'],n['identity'],True)
                self.assertGreater(float((first['clean'][:,1:]-changed['clean'][:,1:]).abs().sum()),0)
            else:self.assertIsNone(row['supervision_file'])
        changed=copy.deepcopy(cfg);changed['calibration']['steps']+=1
        with self.assertRaisesRegex(ValueError,'fresh STATE_NAMESPACE'):runner.checked_manifest(out,changed)
        with patch.object(runner,'code_signature',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'fresh STATE_NAMESPACE'):runner.checked_manifest(out,cfg)

    def test_failed_input_continues_and_is_never_reported_completed(self):
        cfg,args,out=fixture(self.root,broken=True)
        runner.states(args,cfg,out,self.device,0,1)
        rows=out.json('phases/states.json')['records']
        self.assertEqual(sum(r['status']=='completed' for r in rows),3)
        self.assertEqual(sum(r['status']=='failed' for r in rows),1)
        runner.cache(args,cfg,out,self.device,0,1)
        report=out.json('phases/cache.json')
        self.assertEqual(len(report['records']),3);self.assertEqual(len(report['skipped_states']),1)
        args.arm='native';runner.render(args,cfg,out,self.device,0,1)
        cases=out.json('phases/render_native.json')['cases']
        self.assertEqual(sum(r['status']=='completed' for r in cases),1)
        with self.assertRaisesRegex(RuntimeError,'Incomplete Stage65'):runner.summarize(out,cfg)
        summary=out.json('summary.json')
        self.assertEqual(summary['status'],'incomplete')
        self.assertTrue(any(r.get('id')=='dev-bad' for r in summary['failures']))

    def test_prefix_is_conditioning_only_not_video_flow_supervision(self):
        cfg,args,out=self.prepare()
        m=out.json('manifest.json');row=next(r for r in out.json('phases/cache.json')['records'] if r['partition']=='train')
        item=runner.load_case(out,row,m['identity'],'rae',self.device,supervised=True)
        sigma=torch.tensor([.4]);noise=torch.randn_like(item['clean'])[None]
        _,target=runner.flow_pair(item['clean'][None].float(),noise,sigma)
        class PrefixWrong(nn.Module):
            def forward(self,*_):
                output=target.clone();output[:,:,:1]+=1000
                return output
        loss,_=runner.flow_objective(PrefixWrong(),item,sigma,noise,cfg)
        self.assertEqual(float(loss),0.)


if __name__=='__main__':unittest.main()
