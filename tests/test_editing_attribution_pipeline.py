"""Stage70 actual driver/callback/media integration with tiny model fixtures.

Do not stub write_video, samplers, observers, state projectors or artifact IO.
The fixture codecs/weights are deliberately small, not production quality tests.
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
import torch.nn.functional as F

import audit_edit_attribution as run
from utils.scene_run import ArtifactStore
from utils.editing_data import file_hash
from utils.vace_state_data import align_rae_state
from train_vace_edit_state import encoded_reference
from tests.test_editing_state_feedback import Codec, fixture
from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_state_runtime import config
from tests.test_vace_edit_state_pipeline import fixed_backbone

ROOT=Path(__file__).resolve().parents[1]


def decoder(z):
    return F.interpolate(z[0,:,:,:,:3].permute(0,3,1,2),size=(32,32),mode='bilinear').sigmoid()


class AttributionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='attribution validation ',dir=ROOT.parent)
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(cpu_native_import())
        self.console=io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.console))
        torch.manual_seed(71)
        self.out=ArtifactStore(self.root/'out',[str(self.root/'primary'),str(self.root/'mirror')])
        self.p=ArtifactStore(self.root/'parent',[]);self.s=ArtifactStore(self.root/'states',[])
        for name in ('rae.pt','encoder.pt'):(self.root/name).write_bytes(b'fixture')
        self.a=SimpleNamespace(parent=str(self.p.local),parent_root=[],states=str(self.s.local),state_root=[],
            source='',data=str(self.root/'data'),bank=str(self.root/'bank'),bank_root=[],
            rae=str(self.root/'rae.pt'),encoder=str(self.root/'encoder.pt'),model='fixture')
        self.cfg=json.loads((ROOT/'configs/editing_attribution_v1.json').read_text())
        self.cfg.update(smoke=True,sample_steps=6,lpips=False)
        self.cfg['feedback']['solver'].update(sweeps=1,steps=1,joint_weight=0.)
        self.cfg['vae_calibration']['steps']=1
        self.base=dict(config(),state_channels={'rae':514},calibration={'steps':1})
        rae=run.StateConditionedVACE(fixed_backbone(),self.base,514)
        sft=run.ReferenceSFT(fixed_backbone(),self.base)
        rs=run.arm_store(self.s,'rae');ss=run.arm_store(self.p,'sft')
        rs.save('selected.pt',dict(model=rae.deltas()),run.digest(['stage65-fixture','rae']))
        ss.save('selected.pt',dict(model=sft.deltas()),run.digest(['sft-fixture','sft']))
        self.p.write_json('sft_contract.json',dict(identity='sft-fixture'))
        self.pm=dict(identity='parent-fixture',stage65='stage65-fixture',base_config=self.base,
            selected={'rae':rs.json('selected.pt.json')['sha256']},assets={'fixture':True},
            teacher={'rae':file_hash(self.a.rae),'encoder':file_hash(self.a.encoder)})
        self.p.write_json('manifest.json',self.pm)
        row=dict(id='case',group='source-group',source_file='sources/case.pt')
        self.p.write_json('cohort.json',dict(identity='cohort-fixture',ablation_ids=['case'],records=[row]))
        inputs,ref,_,_=fixture('rae_joint')
        inputs.update(source=(inputs['source']*255).round().byte(),
            anchor=torch.full((3,32,32),102,dtype=torch.uint8),
            proposal=torch.full((9,3,32,32),102,dtype=torch.uint8),
            masks=torch.ones(9,1,32,32),confidence=torch.ones(9,1,32,32),
            scale=torch.ones(256),edges=[])
        source_z=inputs['source_z'];raw=source_z+.01;calibrated=source_z+.02
        cal=dict(source_z=source_z,raw=raw,calibrated=calibrated,
            off_state=align_rae_state(source_z,raw,inputs['anchor_z'],inputs,ref),
            on_state=align_rae_state(source_z,calibrated,inputs['anchor_z'],inputs,ref),
            operator_rae_effect=torch.full_like(ref['source'],.04))
        self.p.save('prepared/case.pt',dict(inputs=inputs,reference=encoded_reference(ref),rae_state=cal['on_state']),self.pm['identity'])
        self.p.save('native/case.pt',dict(context=torch.zeros(96,6,4,6)),self.pm['identity'])
        self.p.save('text/case.pt',dict(text=torch.randn(4,12),negative=torch.randn(4,12)),self.pm['identity'])
        self.p.save('sources/case.pt',dict(source=(ref['source']*255).round().byte(),spatial=ref['spatial']),self.pm['identity'])
        ArtifactStore(self.a.bank,[]).write_json('bank.json',{})
        self.stack.enter_context(patch.object(run,'validate_assets',return_value={'fixture':True}))
        self.stack.enter_context(patch.object(run,'load_backbone',side_effect=fixed_backbone))
        self.stack.enter_context(patch.object(run,'load_vae',side_effect=lambda *_:Codec()))
        self.stack.enter_context(patch.object(run,'load_decoder_only',return_value=decoder))
        run.index(self.a,self.cfg,self.out)
        self.m=self.out.json('manifest.json');self.identity=self.m['identity']
        self.out.save('calibration/case.pt',cal,self.identity)
        self.ref=ref;self.device=torch.device('cpu')

    def legacy_contract(self):
        migration=json.loads((ROOT/'configs/editing_attribution_resume_compat.json').read_text())
        old=copy.deepcopy(self.m);old['code'].update(migration['before'])
        old.pop('identity');old['identity']=run.digest(old)
        self.out.write_json('manifest.json',old)
        return old

    def test_exact_repair_preserves_manifest_and_rejects_unreviewed_changes(self):
        old=self.legacy_contract()
        run.index(self.a,self.cfg,self.out)
        self.assertEqual(self.out.json('manifest.json'),old)
        self.assertEqual(run.checked(self.a,self.cfg,self.out),old)
        self.assertTrue(list((self.out.local/'recovery').glob('compat_*.json')))
        for name in ('audit_edit_attribution.py','utils/editing_attribution.py','utils/editing_joint_state.py'):
            changed=run.signature();changed[name]='unreviewed'
            with self.subTest(name=name),patch.object(run,'signature',return_value=changed),self.assertRaisesRegex(ValueError,'contract changed'):
                run.checked(self.a,self.cfg,self.out)
        altered=copy.deepcopy(self.cfg);altered['sample_steps']=30
        with self.assertRaisesRegex(ValueError,'contract changed'):run.checked(self.a,altered,self.out)
        p=copy.deepcopy(self.pm);p['selected']['rae']='different-weight';self.p.write_json('manifest.json',p)
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):run.index(self.a,self.cfg,self.out)

    def test_all_arms_real_preview_publication_evaluation_and_resume(self):
        # Production prepared caches skip encoder/decoder fitting on recovery.
        cached=self.out.local/'calibration/case.pt';cached_sha=file_hash(cached)
        with patch.object(run,'load_teacher',side_effect=AssertionError('Do not refit prepared RAE')):
            run.prepare(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(file_hash(cached),cached_sha)
        run.prepare_vae(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(self.out.json('phases/prepare_vae.json')['completed'],1)
        with patch.object(run,'load_vae',side_effect=AssertionError('Do not refit prepared VAE')):
            run.prepare_vae(self.a,self.cfg,self.out,self.device,0,1)
        outputs={}
        for arm in run.ARMS:
            self.a.arm=arm;run.render(self.a,self.cfg,self.out,self.device,0,1)
            report=self.out.json('phases/render_'+arm+'.json')
            self.assertEqual(report['completed'],1,report)
            folder=f'render/{arm}/seed42/case'
            outputs[arm]=self.out.load(folder+'/result.pt',self.identity,True)
            if arm=='op_sft':continue
            events=outputs[arm]['events'];self.assertTrue(events)
            self.assertTrue(all(e['applied'] and not e['fallback'] for e in events),events)
            for event in events:
                name=folder+f'/feedback/step_{event["step_index"]:03d}.mp4'
                receipt=self.out.json(name.replace('.mp4','.json'),True)
                self.assertEqual(receipt['contract_hash'],self.identity)
                self.assertEqual(len({file_hash(Path(r)/name) for r in [self.out.local,*self.out.roots]}),1)
                reader=imageio.get_reader(str(self.out.local/name))
                try:
                    frame=reader.get_data(0)
                    self.assertEqual(reader.count_frames(),17)
                    self.assertAlmostEqual(reader.get_meta_data()['fps'],8.)
                finally:reader.close()
                # Valid crop is 24x40, with THREE panels and real uint8 pixels.
                self.assertEqual(frame.shape,(24,120,3))
                self.assertGreater(float(frame[-5:,:40].mean()),40.)
        self.assertGreater(float((outputs['op_sft']['generated']-outputs['op_rae']['generated']).abs().max()),0.)
        self.assertEqual(len({p['input_identity'] for p in outputs.values()}),1)
        source=(self.ref['source']*255).round().byte()
        with patch('utils.editing_paper_data.load_evaluation_labels',return_value=source):
            run.evaluate(self.a,self.cfg,self.out,self.device,0,1)
        run.summarize(self.a,self.cfg,self.out)
        self.assertTrue(self.out.json('summary.json')['complete'])
        self.assertEqual(self.out.json('summary.json')['evaluated'],14)
        retained={arm:file_hash(self.out.local/f'render/{arm}/seed42/case/result.pt') for arm in run.ARMS}
        # Simulate one missing preview replica: reconcile it without generation.
        media=next((self.out.local/'render/op_rae/seed42/case/feedback').glob('*.mp4'))
        (Path(self.out.roots[0])/media.relative_to(self.out.local)).unlink()
        with patch.object(run,'load_backbone',side_effect=AssertionError('Do not resample completed cases')):
            for arm in run.ARMS:
                self.a.arm=arm;run.render(self.a,self.cfg,self.out,self.device,0,1)
            run.summarize(self.a,self.cfg,self.out)
        self.assertTrue(self.out.json('summary.json')['complete'])
        self.assertEqual(retained,{arm:file_hash(self.out.local/f'render/{arm}/seed42/case/result.pt') for arm in run.ARMS})
        self.assertEqual(self.out.json('progress/render_op_rae_r0.json')['DI_throughput'],0.)

    def test_repaired_legacy_run_reuses_completed_sft_and_raw_states(self):
        cal=self.out.load('calibration/case.pt',self.identity,True)
        old=self.legacy_contract();identity=old['identity']
        self.out.save('calibration/case.pt',cal,identity)
        run.index(self.a,self.cfg,self.out)
        self.a.arm='op_sft';run.render(self.a,self.cfg,self.out,self.device,0,1)
        folder='render/op_sft/seed42/case'
        retained={name:file_hash(self.out.local/name) for name in
            ('calibration/case.pt',folder+'/result.pt',folder+'/metrics.json',folder+'/receipt.json')}
        generation=self.out.load(folder+'/result.pt',identity,True)['generation_identity']
        with patch.object(run,'load_backbone',side_effect=AssertionError('Do not repeat SFT sampling')),\
             patch.object(run,'load_teacher',side_effect=AssertionError('Do not repeat RAE fitting')):
            run.index(self.a,self.cfg,self.out)
            run.prepare(self.a,self.cfg,self.out,self.device,0,1)
            run.render(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(self.out.json('manifest.json'),old)
        self.assertEqual(retained,{name:file_hash(self.out.local/name) for name in retained})
        self.assertEqual(self.out.load(folder+'/result.pt',identity,True)['generation_identity'],generation)
        self.assertEqual(self.out.json('phases/render_op_sft.json')['completed'],1)
        self.assertEqual(self.out.json('progress/render_op_sft_r0.json')['DI_throughput'],0.)

    def test_summary_exposes_upstream_failure_instead_of_only_missing_media(self):
        error='injected preview writer failure'
        run.finish_phase(self.out,'render_op_rae',[dict(id='case',status='failed',error=error)],1,0,1)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):run.summarize(self.a,self.cfg,self.out)
        summary=self.out.json('summary.json')
        self.assertEqual(summary['failure_causes'][error],1)
        self.assertIn(error,self.console.getvalue())


if __name__=='__main__':unittest.main()
