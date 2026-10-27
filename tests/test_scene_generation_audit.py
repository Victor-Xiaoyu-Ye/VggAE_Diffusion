"""Executable tiny CPU tests; not real checkpoint or NPU quality validation."""
import copy
import json
import unittest

import torch
from torch import nn

from utils.scene_generation_audit import audit_generation_case, _rms_matched_gaussian
from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.scene_prefix_eval import stable_case_seed
from utils.window_flow import WindowFlow


class TinyCodec(nn.Module):
    channels=4
    grid=2
    def __init__(self, use_geo=True):
        super().__init__(); self.use_geo=use_geo
    def decode(self,z):
        texture=z[...,2:].mean(-1)
        geo=z[...,:2].mean(-1) if self.use_geo else 0
        value=(texture+geo).sigmoid()
        return value[...,None].expand(*value.shape,3)


class TinyFlow(nn.Module):
    def __init__(self,controlled=False):
        super().__init__(); self.args={'text_dim':0}
        self.weight=nn.Parameter(torch.tensor(.25))
        if controlled:
            self.control=nn.Sequential(nn.Linear(19,4))
            nn.init.zeros_(self.control[-1].weight);nn.init.zeros_(self.control[-1].bias)
    def forward(self,x,u,anchor,text=None,text_valid=None,ref_present=None,rays=None,controls=None):
        out=x*self.weight+anchor*.1
        if hasattr(self,'control') and controls is not None:
            out=out+self.control(controls)
        return out


def sample():
    generator=torch.Generator().manual_seed(33)
    z=torch.randn(9,4,4,generator=generator)
    raw=TinyCodec().decode(z.reshape(1,9,2,2,4))[0].permute(0,3,1,2)
    points=torch.tensor([[.25,.25],[.75,.75]])[None].expand(9,-1,-1).clone()
    return dict(id='synthetic',frame_ids=list(range(10,100,10)),target=z,raw=raw,
        controls=dict(rays=torch.randn(9,4,8,generator=generator),motion=torch.zeros(9,4,8),
            tracks=points,track_weight=torch.ones(9,2),prefix_tracks=points[:4],prefix_track_weight=torch.ones(4,2)))


class GenerationAuditTests(unittest.TestCase):
    def run_audit(self,model=None,rae=None,item=None,**kwargs):
        return audit_generation_case(model or TinyFlow(),rae or TinyCodec(),item or sample(),
            torch.zeros(4),torch.ones(4),'cpu',steps=2,noise_levels=(.5,),perturb_scales=(.1,),**kwargs)

    def test_contract_fp32_artifacts_and_no_mutation(self):
        model=TinyFlow();rae=TinyCodec(); model.train(); rae.eval()
        state=copy.deepcopy(model.state_dict()); rng=torch.random.get_rng_state().clone()
        report,artifacts=self.run_audit(model,rae)
        self.assertTrue(model.training); self.assertFalse(rae.training)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(state['weight'],model.weight))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(report['frame_ids'],list(range(10,100,10)))
        self.assertEqual(report['sampler']['grid'],[1.,.5,0.])
        self.assertEqual(report['sampler']['state_dtype'],'torch.float32')
        self.assertEqual(tuple(artifacts['preview'].shape),(9,2,8,3))
        self.assertEqual(tuple(artifacts['generated_future_latent'].shape),(5,4,4))
        self.assertTrue(all(r['uses_future_gt'] for r in report['gt_oracles']))
        json.dumps(report,allow_nan=False)

    def test_future_targets_do_not_enter_free_sampling(self):
        item=sample(); altered=copy.deepcopy(item); altered['target'][4:]+=100
        _,a=self.run_audit(item=item,branch_audit=False)
        _,b=self.run_audit(item=altered,branch_audit=False)
        torch.testing.assert_close(a['generated_future_latent'],b['generated_future_latent'],atol=0,rtol=0)

    def test_sampler_matches_production_windowflow(self):
        model=TinyFlow(); item=sample()
        _,artifacts=self.run_audit(model=model,item=item,branch_audit=False)
        batch=prepare_prefix(item,torch.zeros(4),torch.ones(4),hist=4)
        generator=torch.Generator().manual_seed(stable_case_seed(item,101))
        noise=torch.randn(batch.target.shape,generator=generator)
        expected=WindowFlow().sample(PrefixFlowNet(model,batch.known),batch.anchor,noise,steps=2)
        torch.testing.assert_close(artifacts['generated_future_latent'],expected[0],atol=0,rtol=0)

    def test_controlled_sampling_never_uses_future_texture(self):
        model=TinyFlow(True)
        with torch.no_grad(): model.control[-1].weight.fill_(.02)
        item=sample(); changed=copy.deepcopy(item); changed['target'][4:,:,2:]+=100
        _,a=self.run_audit(model=model,item=item,branch_audit=False)
        _,b=self.run_audit(model=model,item=changed,branch_audit=False)
        torch.testing.assert_close(a['generated_future_latent'],b['generated_future_latent'],atol=0,rtol=0)

    def test_decoder_branch_ignored_vs_used(self):
        ignored,_=self.run_audit(rae=TinyCodec(False))
        used,_=self.run_audit(rae=TinyCodec(True))
        ignored_rows=[r for r in ignored['decoder_branches']['rows'] if r['branch']=='geo']
        self.assertTrue(all(r['rgb_change']['max_abs']==0 for r in ignored_rows))
        self.assertTrue(any(r['rgb_change']['max_abs']>0 for r in used['decoder_branches']['rows'] if r['branch']=='geo'))
        self.assertEqual(ignored['decoder_branches']['shuffle'],'within_frame_spatial_permutation')
        structure=ignored['decoder_error_structure']['rows']
        self.assertTrue(all(r['rgb_vs_ae']['max_abs']==0 for r in structure if r['branch']=='geo'))
        self.assertTrue(all(r['rgb_vs_ae']['max_abs']>0 for r in structure if r['branch']=='texture'))

    def test_error_structure_matches_mse_but_not_error_direction(self):
        item=sample()
        mean=torch.tensor([.2,-.4,1.,.3]); std=torch.tensor([2.,.5,3.,.25])
        report,artifacts=audit_generation_case(TinyFlow(),TinyCodec(),item,mean,std,'cpu',
            steps=2,noise_levels=(.5,),perturb_scales=(.1,))
        probe=report['decoder_error_structure']
        self.assertTrue(probe['uses_future_gt'])
        self.assertEqual(probe['other_branch'],'unchanged_future_gt')
        self.assertEqual(len(probe['rows']),4)
        for branch,start,end in (('geo',0,2),('texture',2,4)):
            actual,gaussian=[r for r in probe['rows'] if r['branch']==branch]
            expected=(artifacts['generated_future_latent'][...,start:end]-item['target'][4:,:,start:end])/std[start:end]
            self.assertAlmostEqual(actual['normalized_error_mse'],float(expected.square().mean()),places=6)
            self.assertAlmostEqual(actual['normalized_error_mse'],gaussian['normalized_error_mse'],places=6)
            self.assertAlmostEqual(actual['normalized_error']['rms'],gaussian['normalized_error']['rms'],places=6)
            self.assertEqual(actual['error_vs_actual_generated_error']['max_abs'],0)
            self.assertGreater(gaussian['error_vs_actual_generated_error']['rms'],0)
            self.assertAlmostEqual(gaussian['realized_normalized_error_mse'],gaussian['normalized_error_mse'],places=6)
            self.assertTrue(gaussian['rgb_vs_ae']['all_finite'])

    def test_error_structure_zero_error_and_independent_rng(self):
        class ZeroFlow(TinyFlow):
            def forward(self,x,u,anchor,*args,**kwargs):
                return torch.zeros_like(x)
        item=sample();item['target'].zero_()
        rng=torch.random.get_rng_state().clone()
        report,_=self.run_audit(model=ZeroFlow(),item=item)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for row in report['decoder_error_structure']['rows']:
            self.assertEqual(row['normalized_error_mse'],0)
            self.assertEqual(row['normalized_error']['rms'],0)
            self.assertEqual(row['rgb_vs_ae']['max_abs'],0)
        json.dumps(report,allow_nan=False)
        error=torch.arange(1.,81.).reshape(1,5,4,4)
        first=_rms_matched_gaussian(error,torch.Generator().manual_seed(42))
        second=_rms_matched_gaussian(error,torch.Generator().manual_seed(42))
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        torch.testing.assert_close(first.square().mean(),error.square().mean(),rtol=1e-6,atol=0)
        self.assertFalse(torch.equal(first,error))

    def test_zero_init_adapter_expected_and_active_sensitivity(self):
        model=TinyFlow(True)
        zero,_=self.run_audit(model=model,branch_audit=False)
        self.assertTrue(zero['controls']['zero_init_adapter'])
        self.assertTrue(all(r['rgb_change']['max_abs']==0 for r in zero['controls']['variants']))
        with torch.no_grad(): model.control[-1].weight.fill_(.02)
        active,_=self.run_audit(model=model,branch_audit=False)
        self.assertFalse(active['controls']['zero_init_adapter'])
        self.assertGreater(active['controls']['reference_support_fraction'],0)
        self.assertGreater(active['controls']['variants'][0]['rgb_change']['max_abs'],0)

    def test_cross_scene_donor_and_invalid_layout(self):
        donor=sample(); donor['id']='other'; donor['target']+=3
        report,_=self.run_audit(matched_other_sample=donor)
        self.assertEqual(report['decoder_branches']['shuffle'],'cross_scene')
        self.assertEqual(report['decoder_branches']['donor_id'],'other')
        bad=sample();bad['target']=bad['target'][:8]
        with self.assertRaisesRegex(ValueError,'nine-frame'): self.run_audit(item=bad)

    def test_actual_scene_flow_and_controlled_warm_start_interfaces(self):
        from models.scene_flow import SceneFlow
        from utils.scene_control import ControlledSceneFlow, load_warm
        args=dict(channels=4,grid=2,width=24,depth=1,head_width=48,head_depth=1,
                  heads=4,text_dim=8,checkpoint_blocks=False)
        base=SceneFlow(**args)
        report,_=self.run_audit(model=base,branch_audit=False)
        self.assertEqual(report['controls']['status'],'not_applicable_source_model')
        controlled=ControlledSceneFlow(**args)
        load_warm(controlled,base.state_dict())
        report,_=self.run_audit(model=controlled,branch_audit=False)
        self.assertTrue(report['controls']['zero_init_adapter'])
        self.assertTrue(all(r['latent_change']['max_abs']==0 for r in report['controls']['variants']))

    def test_camera_audit_accepts_actual_controlled_model_and_fixed_unknown_context(self):
        from models.scene_flow import SceneFlow
        from utils.scene_control import create_control_model, load_warm
        args=dict(channels=4,grid=2,width=24,depth=1,head_width=48,head_depth=1,
                  heads=4,text_dim=8,checkpoint_blocks=False)
        model=create_control_model(args,'camera')
        load_warm(model,SceneFlow(**args).state_dict())
        report,artifacts=self.run_audit(model=model,branch_audit=False,control_arm='camera')
        self.assertEqual(report['controls']['arm'],'camera')
        self.assertEqual(report['controls']['track_support_fraction'],0)
        self.assertEqual(report['controls']['reference_support_fraction'],0)
        self.assertEqual(report['controls']['zero_condition'],'camera_rays_zero_with_unknown_one')
        self.assertEqual(tuple(artifacts['generated_future_latent'].shape),(5,4,4))
        self.assertTrue(all(r['latent_change']['max_abs']==0 for r in report['controls']['variants']))


if __name__=='__main__':
    unittest.main()
