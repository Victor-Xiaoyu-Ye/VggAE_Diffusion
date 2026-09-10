import unittest
from types import SimpleNamespace
import torch
from utils.window_flow import WindowFlow
from utils.window_training import resume_contract


class TimeDistributionTests(unittest.TestCase):
    def test_old_identity_and_draws_unchanged(self):
        flow=WindowFlow(time_shift=3)
        expected=dict(schema='r7-window-flow-v1',path='(1-u)*data+u*noise',
                      time_distribution='logit_normal_0_1',prediction='x0',loss_floor=.05,time_shift=3)
        self.assertEqual(flow.contract(),expected)
        torch.manual_seed(7);u=torch.randn(100).sigmoid();expected_u=(3*u/(1+2*u)).clamp(1e-5,1-1e-5)
        torch.manual_seed(7);torch.testing.assert_close(flow.times(100,'cpu'),expected_u,atol=0,rtol=0)
        old=SimpleNamespace(time_shift=3)
        new=SimpleNamespace(time_shift=3,time_distribution='logit_normal_0_1')
        self.assertEqual(resume_contract(old,{},48),resume_contract(new,{},48))

    def test_uniform_bins_and_same_loss_sampler(self):
        uniform=WindowFlow(time_distribution='uniform')
        torch.manual_seed(9);u=uniform.times(100000,'cpu')
        counts=torch.bincount((u*5).long(),minlength=5).float()/len(u)
        self.assertTrue(bool(((counts-.2).abs()<.01).all()))
        self.assertTrue(bool(((u>0)&(u<1)).all()))
        y=torch.randn(1,2,4,6);noise=torch.randn_like(y);c=y[:,:1];t=torch.tensor([.2])
        torch.testing.assert_close(uniform.loss(y+.1,y,noise,t),WindowFlow().loss(y+.1,y,noise,t))
        def model(x,u,c,*args):return .1*x+c
        torch.testing.assert_close(uniform.sample(model,c,noise),WindowFlow(time_shift=3).sample(model,c,noise),atol=0,rtol=0)
        with self.assertRaises(ValueError):WindowFlow(time_shift=3,time_distribution='uniform')


if __name__=='__main__':unittest.main()
