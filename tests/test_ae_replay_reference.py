import copy
import json
from pathlib import Path
import unittest
from utils.ae_replay_reference import compare_replay


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ref=json.loads((Path(__file__).parents[1]/'configs/ae_reference_single_v1.json').read_text())
        self.result=copy.deepcopy(self.ref)

    def test_frozen_baseline_and_rounding_pass_but_regression_fails(self):
        self.result['clips'].reverse()
        self.assertTrue(compare_replay(self.result,self.ref)['passed'])
        self.result['clips'][0]['ae_psnr_full_vs_raw'] += .001
        self.assertTrue(compare_replay(self.result,self.ref)['passed'])
        self.result['clips'][0]['ae_psnr_full_vs_raw'] -= .1
        self.assertFalse(compare_replay(self.result,self.ref)['passed'])

    def test_identity_and_nonfinite_fail_closed(self):
        for case in ('ids','duplicate','signature','norm','nan'):
            with self.subTest(case=case):
                value=copy.deepcopy(self.ref)
                if case=='ids':value['clips'][0]['video_id']='other'
                if case=='duplicate':value['clips'][0]=value['clips'][1]
                if case=='signature':value['ae_signature']={}
                if case=='norm':value['temporal_norm']='framewise'
                if case=='nan':value['clips'][0]['ae_psnr_full_vs_raw']=float('nan')
                with self.assertRaises(ValueError):compare_replay(value,self.ref)
