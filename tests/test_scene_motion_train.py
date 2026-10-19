"""Actual tiny flow integration: new objective, borrowed cache, stopping/resume."""
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_scene_prefix as trainer
from tests.test_scene_prefix_train import (fixture,new_store,seed_all,TinyRAE,
    fake_preview,record_metrics,ddp_worker)
from utils.scene_run import StageBudget


def execute(data,output,manifest,bank,*,stage='train',steps=8,preview_fn=fake_preview):
    cfg=dict(data['cfg'],max_steps=steps,source_label='source_fixture',eval_every_steps=1,
        motion_objective=dict(strength=1.,cap=3.,log_activity=True),
        convergence=dict(min_steps=2,patience=1,relative_tol=.99),oracle_cases=[0])
    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer,'SceneRAE',TinyRAE))
        stack.enter_context(patch.object(trainer,'preview',preview_fn))
        stack.enter_context(patch.object(trainer,'metrics',record_metrics))
        source=data['source']
        trainer.train(SimpleNamespace(stage=stage),cfg,data['plan'],source,output,manifest,data['selection'],
            source.local/'rae.pt',source.local/'weights.pt',source.local/'eval.pt',
            StageBudget(10000),torch.device('cpu'),0,1,data_store=bank)
    return output.load('train/checkpoint_latest.pt',data['plan']['identity'],True)


class MotionTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_borrowed_cache_plateau_resume_and_final_oracles(self):
        from utils.scene_run import ArtifactStore
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=fixture(root)
            bank,manifest,_,_=new_store(root,'bank',data,malformed=False)
            output=ArtifactStore(root/'out',[str(root/'primary'),str(root/'mirror')])
            seed_all(71)
            state=execute(data,output,manifest,bank,steps=2)
            self.assertEqual(state['step'],2)
            self.assertEqual(state['convergence_state']['converged'],True)
            self.assertEqual(output.json('complete.json')['stop_reason'],'optimization_plateau_requires_visual_review')
            self.assertFalse((output.local/'data').exists())
            done=output.json('eval/step0000002/ema/case000_seed101.done.json',True)
            self.assertIn('gt_assisted_diagnostic',done['metadata'])
            self.assertTrue(any('GT_ASSISTED_ORACLE' in f for f in done['files']))
            # Simulate interruption after converged checkpoint, before final completion.
            for destination in [output.local,*[Path(r) for r in output.roots]]:
                (destination/'train/stopped.json').unlink(missing_ok=True)
                (destination/'complete.json').unlink(missing_ok=True)
            output_again=execute(data,output,manifest,bank,steps=2)
            self.assertEqual(output_again['step'],2)
            self.assertEqual(output.json('complete.json')['stop_reason'],'optimization_plateau_requires_visual_review')
            self.assertEqual(output_again['cursors'],state['cursors'])
            for k,v in state['model'].items():
                torch.testing.assert_close(v,output_again['model'][k],rtol=0,atol=0)
            self.assertEqual(output_again['convergence_state'],state['convergence_state'])

    def test_failed_final_oracle_is_repaired_without_training_more(self):
        from utils.scene_run import ArtifactStore
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=fixture(root)
            bank,manifest,_,_=new_store(root,'bank',data,malformed=False)
            output=ArtifactStore(root/'out',[str(root/'primary'),str(root/'mirror')])
            def fail_oracle(store,name,frames):
                if name.startswith('eval/step0000002/') and 'GT_ASSISTED_ORACLE' in name:
                    return []
                if name=='eval/source_fixture/online/case000_seed101.mp4':
                    return []
                return fake_preview(store,name,frames)
            seed_all(71)
            state=execute(data,output,manifest,bank,preview_fn=fail_oracle)
            self.assertEqual(output.json('complete.json')['evaluation_status'],'partial')
            self.assertEqual(output.json('train/stopped.json')['step'],2)
            self.assertEqual(output.json('eval/source_fixture/summary.json')['status'],'partial')
            source_latent=output.load('eval/source_fixture/online/case000_seed101.pt',
                data['plan']['identity'],True)['latent']
            repaired=execute(data,output,manifest,bank)
            self.assertEqual(output.json('complete.json')['evaluation_status'],'completed')
            self.assertEqual(output.json('eval/source_fixture/summary.json')['status'],'completed')
            repaired_source=output.load('eval/source_fixture/online/case000_seed101.pt',
                data['plan']['identity'],True)['latent']
            torch.testing.assert_close(source_latent,repaired_source,rtol=0,atol=0)
            self.assertEqual(repaired['step'],state['step'])
            self.assertEqual(repaired['cursors'],state['cursors'])
            for k,v in state['model'].items():
                torch.testing.assert_close(v,repaired['model'][k],rtol=0,atol=0)
            for arm in ('online','ema'):
                row=output.json(f'eval/step0000002/{arm}/case000_seed101.done.json')['metadata']
                self.assertEqual(row['oracle_preview_status'],'completed')

    def test_motion_objective_two_rank_optimizer_and_checkpoint(self):
        import torch.multiprocessing as mp
        from utils.scene_run import ArtifactStore
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=fixture(root)
            data['cfg'].update(motion_objective=dict(strength=1.,cap=3.,log_activity=True),
                convergence=dict(min_steps=4000,patience=4,relative_tol=.002))
            _,manifest,_,_=new_store(root,'ddp',data,malformed=False)
            torch.save({**{k:v for k,v in data.items() if k!='source'},'manifest':manifest},
                root/'ddp_fixture.pt')
            mp.spawn(ddp_worker,args=(str(root),),nprocs=2,join=True)
            states=[torch.load(str(root/f'rank{rank}_model.pt'),weights_only=False) for rank in range(2)]
            for k,v in states[0].items():
                torch.testing.assert_close(v,states[1][k],rtol=0,atol=0)
            saved=ArtifactStore(root/'ddp',[]).load('train/checkpoint_latest.pt',data['plan']['identity'],True)
            self.assertEqual(saved['step'],2)
            self.assertEqual(saved['cursors'],[4,4])
            self.assertFalse(saved['convergence_state']['converged'])
            self.assertTrue(all(v.dtype==torch.float32 for v in saved['ema'].values()))

    def test_new_launcher_sequential_arms_and_failure_propagation(self):
        bash=Path(r'C:\Program Files\Git\bin\bash.exe')
        if not bash.is_file():
            self.skipTest('Windows Git Bash fixture')
        repo=Path(__file__).resolve().parents[1]
        for failure in (0,7):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=repo.parent) as tmp:
                root=Path(tmp); scripts=root/'scripts/scale'; scripts.mkdir(parents=True)
                (scripts/'54_train_scene_motion_converge.sh').write_bytes(
                    (repo/'scripts/scale/54_train_scene_motion_converge.sh').read_bytes().replace(b'\r\n',b'\n'))
                stub='''#!/bin/bash
printf '%s|%s|%s|%s|%s\\n' "$SCENE_MOTION_ARM" "$SCENE_PREFIX_NAMESPACE" "$SCENE_PREFIX_ENTRYPOINT" "$SCENE_PREFIX_SOURCE_SUFFIX" "$SCENE_PREFIX_NEEDS_ENCODER" >> "$RECORD_FILE"
exit "$FAIL_CODE"
'''
                (scripts/'53_train_scene_prefix_48h.sh').write_text(stub,encoding='utf8',newline='\n')
                env=dict(os.environ,RECORD_FILE=(root/'calls').as_posix(),FAIL_CODE=str(failure),
                    SCENE_MOTION_ARMS='control motion',SCENE_MOTION_NAMESPACE='fixture')
                result=subprocess.run([str(bash),(scripts/'54_train_scene_motion_converge.sh').as_posix()],
                    env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,failure,result.stdout+result.stderr)
                rows=(root/'calls').read_text().splitlines()
                self.assertEqual(len(rows),1 if failure else 2)
                self.assertEqual(rows[0],'control|fixture_control|train_scene_motion.py||0')
                if not failure:
                    self.assertEqual(rows[1],'motion|fixture_motion|train_scene_motion.py||0')


if __name__=='__main__':
    unittest.main()
