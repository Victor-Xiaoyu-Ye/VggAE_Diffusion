import copy
import json
from pathlib import Path
import unittest
import torch

from utils.editing_attribution import NativeEffectFeedback
from utils.editing_readout_strength import (ObservedEffectFeedback,select_cohorts,specifications,
    records_for,arms_for,strength_metrics,matched_strength,validate_config)
from tests.test_editing_state_feedback import Codec,fixture

ROOT=Path(__file__).resolve().parents[1]


class StrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def cfg(self):return json.loads((ROOT/'configs/editing_readout_strength_v1.json').read_text())

    def test_cohort_123_32_91_and_no_duplicates(self):
        cfg=self.cfg();validate_config(cfg)
        rows=[dict(id=str(i),group='g'+str(i)) for i in range(123)]
        cohort=dict(records=rows,ablation_ids=[str(i) for i in range(32)])
        subset,extension=select_cohorts(cohort,cfg)
        self.assertEqual((len(subset),len(extension)),(32,91))
        m=dict(config=cfg,records=rows,subset_ids=cohort['ablation_ids'])
        self.assertEqual(sum(len(arms_for(m,r)) for r in rows),871)
        self.assertEqual(len(records_for(m,'all','op_rae_a015')),32)
        self.assertEqual(records_for(m,'extension','op_rae_a055'),[])
        self.assertEqual(len(records_for(m,'extension','op_rae')),91)
        cfg['smoke']=True
        self.assertEqual([len(v) for v in select_cohorts(cohort,cfg)],[3,3])
        cohort['ablation_ids'][0]='1'
        with self.assertRaises(ValueError):select_cohorts(cohort,cfg)

    def test_observer_keeps_exact_update_and_reports_post_cap_future_dose(self):
        _,ref,_,_=fixture();cfg=self.cfg()['feedback'];cfg['writeback_max_clean_rms']=.001
        effect=ref['reference']-ref['source'];events=[]
        base=NativeEffectFeedback(Codec(),ref,effect,cfg)
        seen=ObservedEffectFeedback(Codec(),ref,effect,cfg,lambda e:events.append(dict(e)))
        for cb in (base,seen):cb.set_schedule([1.,.6,.35,.15,0.])
        x=torch.randn(16,6,4,6)*.1;v=torch.randn_like(x)*.02
        wanted=base(step_index=1,sigma=.6,latent=x,velocity=v)
        actual=seen(step_index=1,sigma=.6,latent=x,velocity=v)
        self.assertTrue(torch.equal(wanted,actual))
        event=events[0];self.assertTrue(event['applied']);self.assertEqual(event['reference_prefix_delta_rms'],0.)
        rms=float(((v-actual)*.6)[:,2:].square().mean().sqrt())
        self.assertAlmostEqual(event['clean_delta_future_rms'],rms)
        self.assertLessEqual(rms,.001001)
        self.assertEqual(len(events),1)
        seen(step_index=1,sigma=.6,latent=x,velocity=v)
        self.assertEqual(len(events),1)

    def test_final_strength_excludes_anchor_and_padding(self):
        _,ref,_,_=fixture();ref=copy.deepcopy(ref)
        ref['masks']=torch.ones_like(ref['masks']);ref['confidence']=torch.ones_like(ref['confidence'])
        ref['reference']=ref['source']+.1
        changed=ref['source'].clone();changed[0]=1
        metrics=strength_metrics(changed,ref)
        self.assertEqual(metrics['future_edit_rms'],0.)
        self.assertEqual(metrics['future_reference_projection'],0.)
        changed[1:]+= .05
        metrics=strength_metrics(changed,ref)
        self.assertAlmostEqual(metrics['future_edit_rms'],.05,places=6)
        self.assertAlmostEqual(metrics['future_reference_projection'],.5,places=5)

    def curves(self):
        records=[]
        for arm,spec in specifications(self.cfg()).items():
            if arm=='op_sft':continue
            alpha=spec['alpha'];rae=spec['operator']=='op_rae'
            records.append(dict(id='case',group='g',seed_base=42,arm=arm,status='completed',metrics=dict(
                future_edit_rms=alpha+(0 if rae else .05),future_reference_projection=alpha,
                target_l1=.2+alpha-(.05 if rae else 0))))
        return records

    def test_matched_strength_midpoint_no_quality_selection(self):
        records=self.curves();report=matched_strength(records,self.cfg())
        control=report['comparisons']['future_edit_rms']['op_rgb']
        self.assertEqual(control['matched'],1)
        self.assertAlmostEqual(control['points'][0]['query'],.375)
        self.assertAlmostEqual(control['points'][0]['values']['rae_full']['future_edit_rms'],.375)
        self.assertAlmostEqual(control['points'][0]['values']['op_rgb']['future_edit_rms'],.375)
        for row in records:row['metrics']['target_l1']*=100
        second=matched_strength(records,self.cfg())
        self.assertEqual(second['comparisons']['future_edit_rms']['op_rgb']['points'][0]['query'],.375)

    def test_no_extrapolation_nonmonotonic_and_missing_are_disclosed(self):
        records=self.curves()
        for row in records:
            if row['arm'].startswith('op_rgb_') or row['arm']=='op_rgb':row['metrics']['future_edit_rms']+=2
            if row['arm']=='op_vae':row['metrics']['future_edit_rms']=9
        records=[r for r in records if r['arm']!='op_rgb_smooth_a015']
        report=matched_strength(records,self.cfg())['comparisons']['future_edit_rms']
        self.assertEqual(report['op_rgb']['excluded'][0]['reason'],'no_common_strength_interval')
        self.assertEqual(report['op_vae']['excluded'][0]['reason'],'flat_or_nonmonotonic_strength')
        self.assertEqual(report['op_rgb_smooth']['excluded'][0]['reason'],'missing_alpha')
        self.assertTrue(all(v['matched']==0 for v in report.values()))

    def test_missing_whole_source_stays_in_strength_denominator(self):
        report=matched_strength(self.curves(),self.cfg(),[dict(id='case',group='g'),dict(id='missing',group='h')])
        comparison=report['comparisons']['future_edit_rms']['op_rgb']
        self.assertEqual(comparison['eligible'],2)
        self.assertEqual(comparison['matched'],1)
        self.assertEqual(comparison['excluded'][0]['id'],'missing')


if __name__=='__main__':unittest.main()
