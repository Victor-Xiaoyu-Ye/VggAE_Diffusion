"""Missing/corrupt output media must be repaired from the durable mirror."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from utils.scene_run import ArtifactStore
from utils.editing_data import digest
from utils.native_audit_io import seal
from utils.editing_publication import reconcile_case,reconcile_evaluations
from scripts.finalize_vace_edit import paired_summary

REPO=Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_evaluate_checkpoint_does_not_enter_optimizer_training(self):
        import train_vace_edit as trainer
        from tests.test_vace_edit_pilot import config
        with tempfile.TemporaryDirectory(prefix='editing eval ',dir=REPO.parent) as temp:
            root=Path(temp);cfg=config();cfgpath=root/'config.json';cfgpath.write_text(json.dumps(cfg))
            data=ArtifactStore(root/'data',[]);out=ArtifactStore(root/'out',[])
            bank=dict(identity='bank',config=cfg,model_signature='assets',records=[])
            data.write_json('bank.json',bank)
            identity=digest(dict(schema='vace-edit-train-v1',cfg=cfg,arm='sft',bank='bank',teacher=None,world=1))
            out.save('checkpoints/step_000002.pt',dict(model={'raw':1},ema={'ema':2},step=2),identity)
            class Model:
                def __init__(self):self.restored=[]
                def to(self,_):return self
                def restore(self,state):self.restored.append(state)
            model=Model()
            args=['train_vace_edit.py','--config',str(cfgpath),'--model','unused','--data',str(data.local),
                  '--output',str(out.local),'--arm','sft','--eval-checkpoint','checkpoints/step_000002.pt']
            with patch('sys.argv',args),patch.object(trainer,'distributed',return_value=(torch.device('cpu'),0,1)),\
                 patch.object(trainer,'validate_assets',return_value='assets'),patch.object(trainer,'load_backbone'),\
                 patch.object(trainer,'EditingModel',return_value=model),\
                 patch.object(trainer,'run_training',side_effect=AssertionError('must not train')),\
                 patch.object(trainer,'evaluate',return_value=[dict(id='a')]) as evaluate:
                trainer.main()
            self.assertEqual(model.restored,[{'raw':1},{'ema':2}])
            self.assertEqual([c.args[10] for c in evaluate.call_args_list],['raw','ema'])

    def test_resume_republishes_without_loading_model_or_sampling(self):
        import train_vace_edit as trainer
        with tempfile.TemporaryDirectory(prefix='editing media ',dir=REPO.parent) as temp:
            root=Path(temp);mirror=root/'mirror';mirror.mkdir()
            folder='samples/s000010/ema/case';movie=folder+'/source_target_output.mp4'
            (mirror/folder).mkdir(parents=True);(mirror/movie).write_bytes(b'complete-video-payload')
            row=dict(id='case',step=10,variant='ema',split='dev',target_psnr=20.,protocol='text_only')
            seal(mirror,folder+'/receipt.json',digest(['run',10,'ema','case']),[movie],row)
            out=ArtifactStore(root/'newlocal',[str(root/'newjob'),str(mirror)])
            model=SimpleNamespace(eval=lambda:None)
            with patch.object(trainer,'load_vae',side_effect=AssertionError('must not load model')):
                result=trainer.evaluate(SimpleNamespace(arm='sft'),dict(steps=10,dev_pairs=1),
                    dict(records=[dict(id='case',split='dev')]),None,out,model,None,0,1,10,'ema','run')
            self.assertEqual(result,[row])
            self.assertEqual((root/'newjob'/movie).read_bytes(),b'complete-video-payload')
            self.assertTrue((root/'newjob'/folder/'receipt.json').is_file())

    def test_reconcile_all_early_steps_and_corruption_not_just_latest(self):
        with tempfile.TemporaryDirectory(prefix='editing media ',dir=REPO.parent) as temp:
            root=Path(temp);mirror=ArtifactStore(root/'mirror',[]);primary=root/'primary'
            for step in (2,250):
                folder=f'samples/s{step:06d}/raw/id';movie=folder+'/source_target_output.mp4'
                (mirror.local/folder).mkdir(parents=True);(mirror.local/movie).write_bytes(f'video-{step}'.encode())
                row=dict(id='id',step=step,variant='raw',split='dev')
                seal(mirror.local,folder+'/receipt.json','contract'+str(step),[movie],row)
                mirror.write_json(f'evaluation/s{step:06d}_raw.json',dict(cases=[row],protocol='text_only'))
                (primary/folder).mkdir(parents=True);(primary/movie).write_bytes(b'truncated')
            store=ArtifactStore(root/'fresh',[str(primary),str(mirror.local)])
            report,values=reconcile_evaluations(store)
            self.assertEqual(report['status'],'synced');self.assertEqual(report['checked'],2)
            self.assertEqual(len(values),2)
            for step in (2,250):
                self.assertEqual((primary/f'samples/s{step:06d}/raw/id/source_target_output.mp4').read_bytes(),f'video-{step}'.encode())
            report,_=reconcile_evaluations(store)
            self.assertTrue(all(not r['repaired'] for c in report['cases'] for r in c['roots']))

    def test_dual_failure_never_marked_synced_and_receipt_follows_payload(self):
        with tempfile.TemporaryDirectory(prefix='editing media ',dir=REPO.parent) as temp:
            root=Path(temp);store=ArtifactStore(root/'local',[str(root/'broken'),str(root/'mirror')])
            (store.local/'movie.mp4').write_bytes(b'video')
            seal(store.local,'receipt.json','id',['movie.mp4'],{})
            from utils.editing_publication import copy_file
            order=[]
            def copy(source,destination):
                order.append(destination)
                if 'broken' in destination:raise OSError('temporary output unavailable')
                copy_file(source,destination)
            with patch('utils.editing_publication.copy_file',side_effect=copy):
                report=reconcile_case(store,'receipt.json','id',attempts=2,delay=0)
            self.assertFalse(report[0]['ok']);self.assertTrue(report[1]['ok'])
            payload=str(root/'mirror'/'movie.mp4').replace('\\','/')
            receipt=str(root/'mirror'/'receipt.json').replace('\\','/')
            normalized=[p.replace('\\','/') for p in order]
            self.assertLess(normalized.index(payload),normalized.index(receipt))

    def test_paired_summary_rejects_mismatched_conditions_and_missing_cases(self):
        a=dict(id='a',split='dev',seed=1,target_psnr=20.)
        b=dict(id='b',split='dev',seed=1,target_psnr=10.)
        left=dict(cases=[a,b],failed=0)
        right=dict(cases=[dict(a,seed=2),dict(b,target_psnr=12.),dict(id='c',error='failed')],failed=1)
        report=paired_summary({'sft':{'evaluation/s000010_ema.json':left},'rae':{'evaluation/s000010_ema.json':right}})
        pair=next(p for p in report['comparisons'] if p['method']=='rae')
        self.assertEqual(pair['paired'],1);self.assertEqual(pair['mismatched_conditions'],['a'])
        self.assertEqual(pair['metrics']['target_psnr']['delta_method_minus_baseline'],2.)
        self.assertEqual(pair['method_failures'],1)


if __name__=='__main__':unittest.main()
