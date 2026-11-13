"""Actual tiny VACE/UniPC, frozen state calibration, media and resume.

Only unavailable pretrained models/codecs are replaced. The public-format
decode, geometry, source flow, solver, writeback and four-arm report are real.
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

import imageio.v2 as imageio
import torch
from torch import nn

import audit_public_editing as run
from utils.scene_run import ArtifactStore
from utils.editing_public_report import summarize
from utils.editing_data import file_hash, rae_canvas
from tests.test_editing_public_data import make_benchmark, GEOMETRY
from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_edit_state_pipeline import fixed_backbone, TinyCodec, encode_rae
from tests.test_vace_state_data import TinyRAE

ROOT=Path(__file__).resolve().parents[1]


class PublicPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='public pipeline ',dir=ROOT.parent)
        self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        stack=contextlib.ExitStack(); self.addCleanup(stack.close); self.stack=stack
        stack.enter_context(cpu_native_import())
        self.console=io.StringIO(); stack.enter_context(contextlib.redirect_stdout(self.console))
        self.out=ArtifactStore(self.root/'out',[str(self.root/'primary'),str(self.root/'mirror')])
        self.p=ArtifactStore(self.root/'parent',[]); self.s=ArtifactStore(self.root/'states',[])
        for name in ('rae.pt','encoder.pt'): (self.root/name).write_bytes(b'explicit-test-codec')
        self.a=SimpleNamespace(parent=str(self.p.local),parent_root=[],states=str(self.s.local),state_root=[],
            transport=str(self.root/'transport'),transport_root=[],bank=str(self.root/'bank'),bank_root=[],
            benchmark=str(self.root/'benchmark'),benchmark_source='',model='test-native',
            rae=str(self.root/'rae.pt'),encoder=str(self.root/'encoder.pt'))
        self.raw=make_benchmark(self.a.benchmark)
        self.cfg=json.loads((ROOT/'configs/editing_public_colorization_v1.json').read_text())
        self.cfg.update(smoke=True,cases=3,sample_steps=6,lpips=False)
        self.cfg['feedback']['solver'].update(sweeps=1,steps=1)
        base=json.loads((ROOT/'configs/vace_edit_state_v1.json').read_text())
        base.update(lora_rank=2,lora_alpha=2,bridge_width=8,guidance=2.,support_dilation=1)
        base['transport'].update(flow_backend='farneback',flow_direct_anchor=False)
        base['calibration']['steps']=1
        checks={}
        for family in ('sft','rgb','rae'):
            with torch.random.fork_rng():
                torch.manual_seed(51)
                model=(run.ReferenceSFT(fixed_backbone(),base) if family=='sft' else
                    run.StateConditionedVACE(fixed_backbone(),base,base['state_channels'][family]))
            store=run.arm_store(self.p if family=='sft' else self.s,family)
            store.save('selected.pt',dict(model=model.deltas()),run.digest(['sft-fixture' if family=='sft' else 'state-fixture',family]))
            checks[family]=store.json('selected.pt.json')['sha256']
        run.arm_store(self.p,'sft').write_json('training_summary.json',dict(status='completed',updates=10))
        self.p.write_json('sft_contract.json',dict(identity='sft-fixture'))
        self.pm=dict(identity='parent-fixture',bank='bank-fixture',stage64='transport-fixture',stage65='state-fixture',
            source_geometry=GEOMETRY,base_config=base,selected=checks,assets={'fixture':True},negative_prompt='bad',
            teacher={k:file_hash(getattr(self.a,k)) for k in ('rae','encoder')})
        self.p.write_json('manifest.json',self.pm)
        self.p.write_json('cohort.json',dict(identity='parent-cohort',records=[]))
        self.p.write_json('phases/screen.json',dict(records=[]))
        ArtifactStore(self.a.bank,[]).write_json('bank.json',dict(identity='bank-fixture',records=[]))
        t=ArtifactStore(self.a.transport,[])
        t.write_json('manifest.json',dict(identity='transport-fixture'))
        t.write_json('prepared.json',dict(source_scale=[1.]*256))
        stack.enter_context(patch.object(run,'validate_assets',return_value={'fixture':True}))
        stack.enter_context(patch.object(run,'load_backbone',side_effect=fixed_backbone))
        stack.enter_context(patch.object(run,'load_vae',side_effect=lambda *_:TinyCodec()))
        stack.enter_context(patch.object(run,'load_teacher',side_effect=lambda *_:TinyRAE()))
        stack.enter_context(patch.object(run,'load_encoder',return_value=nn.Identity()))
        stack.enter_context(patch.object(run,'load_text',return_value=lambda prompts,device:[torch.linspace(-.1,.3,48).reshape(4,12) for _ in prompts]))
        stack.enter_context(patch('audit_edit_transport.encode_window',side_effect=encode_rae))
        stack.enter_context(patch.object(run,'load_decoder_only',side_effect=lambda *_:lambda z:TinyRAE().decode(z)[:,0].permute(0,3,1,2)))
        small_canvas=lambda frames,rect,**_:rae_canvas(frames,rect,size=32)
        stack.enter_context(patch.object(run,'rae_canvas',side_effect=small_canvas))
        stack.enter_context(patch('utils.editing_public_data.rae_canvas',side_effect=small_canvas))
        run.index(self.a,self.cfg,self.out)
        self.m=self.out.json('manifest.json'); self.device=torch.device('cpu')

    def phase(self,name):
        getattr(run,name)(self.a,self.cfg,self.out,self.device,0,1)
        key='text' if name=='texts' else ('render_'+self.a.arm if name=='render' else name)
        result=self.out.json('phases/'+key+'.json')
        self.assertEqual(result['completed'],result['requested'],result)

    def inputs(self):
        self.phase('sources'); run.select(self.a,self.cfg,self.out)
        for phase in ('texts','anchors','shared','cache'): self.phase(phase)

    def test_all_four_arms_target_firewall_real_feedback_and_resume(self):
        target=Path(self.a.benchmark)/self.raw[0]['raw_video']; data=target.read_bytes(); target.unlink()
        with patch.object(run,'read_target',side_effect=AssertionError('GT must not enter generation')):
            self.inputs(); run.weights(self.a,self.cfg,self.out)
            # Three independent arms finish before any RAE preparation exists.
            for arm in ('native','sft','rgb_full'):
                self.a.arm=arm; self.phase('render')
            self.phase('prepare'); self.a.arm='rae_full'; self.phase('render')
        target.write_bytes(data)
        key=run.cohort(self.out)[0]['id']; outputs={}
        for arm in run.ARMS:
            folder=f'render/{arm}/seed42/{key}'
            payload=self.out.load(folder+'/result.pt',self.m['identity'],True); outputs[arm]=payload
            self.assertEqual(tuple(payload['generated'].shape),(17,3,16,32))
            self.assertTrue((self.out.local/folder/'generated_frames.npz').exists())
            if arm.endswith('_full'):
                self.assertTrue(payload['events'])
                self.assertTrue(all(not e.get('fallback') for e in payload['events']),payload['events'])
                events=[e for e in payload['events'] if e.get('applied')]; self.assertTrue(events)
                movie=self.out.local/folder/f'feedback/step_{int(events[0]["step_index"]):03d}.mp4'
                reader=imageio.get_reader(str(movie))
                try:
                    self.assertEqual(reader.count_frames(),17)
                    frame=reader.get_data(0); self.assertEqual(frame.shape,(16,96,3))
                    self.assertGreater(float(frame[-4:,:32].mean()),10.)
                finally: reader.close()
        self.assertEqual(len({p['input_identity'] for p in outputs.values()}),1)
        self.assertGreater(float((outputs['native']['generated']-outputs['rae_full']['generated']).abs().max()),0.)
        self.phase('evaluate'); summarize(self.a,self.cfg,self.out)
        self.assertTrue(self.out.json('summary.json')['complete'])
        self.assertEqual(self.out.json('summary.json')['evaluated'],4)
        self.assertEqual(self.out.json('media_sync.json')['failed'],0)
        retained={arm:file_hash(self.out.local/f'render/{arm}/seed42/{key}/result.pt') for arm in run.ARMS}
        movie=next((self.out.local/'render/rae_full').rglob('step_*.mp4'))
        (Path(self.out.roots[0])/movie.relative_to(self.out.local)).unlink()
        with patch.object(run,'load_backbone',side_effect=AssertionError('No model loading on completed recovery')),\
             patch.object(run,'load_teacher',side_effect=AssertionError('No repeated calibration')):
            self.phase('prepare')
            for arm in run.ARMS: self.a.arm=arm; self.phase('render')
            summarize(self.a,self.cfg,self.out)
        self.assertEqual(self.out.json('progress/render_rae_full_r0.json')['DI_throughput'],0.)
        self.assertEqual(retained,{arm:file_hash(self.out.local/f'render/{arm}/seed42/{key}/result.pt') for arm in run.ARMS})
        self.assertEqual(file_hash(movie),file_hash(Path(self.out.roots[0])/movie.relative_to(self.out.local)))
        # A fresh worker recovers the exact states/videos from durable roots.
        previous=self.out; self.out=ArtifactStore(self.root/'fresh',previous.roots)
        with patch.object(run,'load_backbone',side_effect=AssertionError('No resampling on a fresh worker')),\
             patch.object(run,'load_teacher',side_effect=AssertionError('No refitting on a fresh worker')):
            run.index(self.a,self.cfg,self.out)
            self.inputs(); self.phase('prepare')
            for arm in run.ARMS: self.a.arm=arm; self.phase('render')
            summarize(self.a,self.cfg,self.out)
        self.assertEqual(retained,{arm:file_hash(self.out.local/f'render/{arm}/seed42/{key}/result.pt') for arm in run.ARMS})

    def test_missing_rae_is_reported_without_preventing_other_arms(self):
        self.inputs()
        for arm in ('native','sft','rgb_full'): self.a.arm=arm; self.phase('render')
        self.a.arm='rae_full'; run.render(self.a,self.cfg,self.out,self.device,0,1)
        report=self.out.json('phases/render_rae_full.json'); self.assertEqual(report['completed'],0)
        run.evaluate(self.a,self.cfg,self.out,self.device,0,1)
        with self.assertRaisesRegex(RuntimeError,'incomplete'): summarize(self.a,self.cfg,self.out)
        result=self.out.json('summary.json')
        self.assertEqual(result['coverage']['native']['completed'],1)
        self.assertEqual(result['coverage']['rae_full']['missing'],[[run.cohort(self.out)[0]['id'],42]])
        self.assertEqual(result['tables']['matched']['means'],{})
        self.assertTrue(result['failure_causes'])

    def test_config_weights_and_all_three_node_assignment(self):
        before=copy.deepcopy(self.cfg); before['sample_steps']=30
        with self.assertRaisesRegex(ValueError,'changed'): run.checked(self.a,before,self.out)
        pm=copy.deepcopy(self.pm); pm['selected']['rgb']='wrong'; self.p.write_json('manifest.json',pm)
        with self.assertRaisesRegex(ValueError,'checkpoint changed'): run.index(self.a,self.cfg,self.out)
        owners={i:r for r in range(24) for i in run.assignments(list(range(20)),r,24)}
        self.assertEqual(set(owners),set(range(20)))
        self.assertEqual([owners[i] for i in range(3)],[0,8,16])
        self.assertEqual(len(set(owners.values())),20)


if __name__=='__main__': unittest.main()
