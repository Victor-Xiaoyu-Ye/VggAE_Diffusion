import unittest
from argparse import Namespace
import torch
from models.r7_window_dit import R7WindowDiT
from utils.window_training import resume_contract


class AuxiliaryTests(unittest.TestCase):
    def test_initialization_gradient_path_and_inference(self):
        torch.set_num_threads(1)
        kwargs=dict(channels=6, grid=2, future=2, width=24, depth=3, heads=3, text_dim=0)
        torch.manual_seed(42); plain=R7WindowDiT(**kwargs)
        rng=torch.random.get_rng_state()
        torch.manual_seed(42); aux=R7WindowDiT(**kwargs, aux_layer=1)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for key,value in plain.state_dict().items():
            torch.testing.assert_close(value,aux.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(1,2,4,6); c=torch.randn(1,1,4,6); u=torch.tensor([.5])
        # Nonzero head weights expose the gradient route past zero initialization.
        with torch.no_grad(): aux.aux_head[-1].weight.normal_(std=.01)
        main,mid=aux(x,u,c,return_aux=True)
        mid.square().mean().backward()
        self.assertGreater(float(aux.input.weight.grad.abs().sum()),0)
        self.assertIsNone(aux.blocks[1].ff[0].weight.grad)
        self.assertIsNone(aux.head.weight.grad)
        aux.eval()
        torch.testing.assert_close(aux(x,u,c), main)
        with torch.no_grad(): aux.aux_head[-1].weight.fill_(100)
        torch.testing.assert_close(aux(x,u,c), main)

    def test_disabled_contract_compatibility(self):
        old=Namespace(depth=12)
        new=Namespace(depth=12,aux_layer=0,aux_weight=0.)
        self.assertEqual(resume_contract(old,{},48),resume_contract(new,{},48))
