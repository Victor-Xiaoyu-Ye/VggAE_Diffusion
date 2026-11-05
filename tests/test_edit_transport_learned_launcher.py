"""Exercise the real Stage64 Bash script with bounded process/cluster stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_scene_matched_launcher import BASH, shell

REPO=Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH, 'Bash required')
class LearnedTransportLauncherTests(unittest.TestCase):
    def run_fixture(self,stage='all',fail='',extra=None,resume_stage=None):
        with tempfile.TemporaryDirectory(prefix='learned transport shell ',dir=REPO.parent) as temp:
            root=Path(temp)
            for directory in ('scripts/scale','scripts/lib','configs'):
                (root/directory).mkdir(parents=True)
            for name in ('scripts/scale/64_train_edit_transport.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/edit_transport_learned_v1.json'):
                shutil.copyfile(REPO/name,root/name)
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
''',encoding='utf8')
            shell(root/'scripts/spatialvid_config.sh','''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output"
SCALE_REMOTE_ROOT="${PROJECT}/primary"
SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache"
VGGAE_REF_ROOT="${PROJECT}/refs"
PERSISTENT_OBS_ROOT="obs://fixture/owner"
STREAMVGGT_CKPT="${PROJECT}/refs/stream.pt"
PYTHON_BIN="${PROJECT}/python-stub.sh"
TORCHRUN_BIN="${PROJECT}/torchrun-stub.sh"
export PROJECT
''')
            shell(root/'scripts/lib/modelarts.sh','''
configure_modelarts_distributed() {
 NNODES=3;NUM_NPUS=8;NODE_RANK=0;WORLD_SIZE=24;MASTER_ADDR=127.0.0.1
 export NNODES NUM_NPUS NODE_RANK WORLD_SIZE MASTER_ADDR
}
ensure_local_checkpoint() {
 echo asset >> "${PROJECT}/calls"
 [[ "${FAIL_PHASE}" != asset ]] || return 7
}
''')
            shell(root/'torchrun-stub.sh','''#!/bin/bash
while [[ "$1" != -- ]];do shift;done;shift
"${PROJECT}/python-stub.sh" "$@"
''')
            shell(root/'python-stub.sh','''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  echo guard-start >> "${PROJECT}/calls"
  echo "guard $*"
  trap 'echo guard-stop >> "${PROJECT}/calls"; exit 0' TERM
  while true;do sleep .05;done ;;
 */window_run_io.py)
  if [[ "$*" == *--watch* ]];then
   trap 'exit 0' TERM
   while true;do sleep .05;done
  fi
  echo publish >> "${PROJECT}/calls" ;;
 */coordinated_scene_controls.py)
  echo "${MASTER_PORT}" >> "${PROJECT}/ports"
  echo "coordinator $*"
  while [[ "$1" != -- ]];do shift;done;shift
  "$@" ;;
 -) "$REAL_PYTHON" "$@" ;;
 */train_edit_transport.py)
  printf 'arg=%s\n' "$@" >> "${PROJECT}/arguments"
  while [[ "$1" != --phase ]];do shift;done;shift
  echo "$1" >> "${PROJECT}/calls"
  sleep .05
  [[ "${FAIL_PHASE}" != "$1" ]] || exit 7 ;;
 *) echo "Unexpected command $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('LEARNED_','EDIT_'))}
            env.update(LEARNED_STAGE=stage,LEARNED_NAMESPACE='fixture',FAIL_PHASE=fail,
                REAL_PYTHON=Path(sys.executable).as_posix(),SCENE_HELPER_STOP_TIMEOUT='1')
            if extra:env.update(extra)
            command=[BASH,(root/'scripts/scale/64_train_edit_transport.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            ns='fixture_smoke' if stage=='smoke' else 'fixture'
            config_path=root/'output'/ns/'launcher/node0/effective_config.json'
            cfg=json.loads(config_path.read_text()) if config_path.exists() else None
            second=None
            if resume_stage:
                env.update(LEARNED_STAGE=resume_stage,LEARNED_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
                self.assertEqual(cfg,json.loads(config_path.read_text()),'Resume overwrote effective configuration')
            def lines(name):
                p=root/name
                return p.read_text().splitlines() if p.exists() else []
            return result,lines('calls'),lines('ports'),cfg,lines('arguments'),second

    def test_full_pipeline_dual_roots_and_readonly_reference_reuse(self):
        result,calls,ports,cfg,args,_=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        phases=[c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases,['asset','asset','prepare','train','predict','render','summarize'])
        self.assertLess(calls.index('guard-start'),calls.index('prepare'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertEqual(len(ports),6)
        self.assertEqual(len(set(ports)),6)
        self.assertIn('--devices 8',result.stdout)
        self.assertIn('--nproc_per_node=8',result.stdout)
        self.assertIn('obs://fixture/owner/cache_latents/vace_edit_pilot_v1',result.stdout)
        self.assertTrue(any('/primary/fixture' in a for a in args))
        self.assertTrue(any('/mirror/fixture' in a for a in args))
        self.assertTrue(any('/output/edit_reference_video_v1' in a for a in args))
        self.assertEqual(args.count('arg=--reference-root'),10)
        self.assertEqual(cfg['train_pairs'],0)
        self.assertEqual(cfg['training']['max_steps'],3000)
        self.assertNotIn('arg=--adapter',args)

    def test_phase_failure_retains_render_summary_and_nonzero_exit(self):
        for failed in ('prepare','train','predict','asset'):
            with self.subTest(failed=failed):
                result,calls,_,_,_,_=self.run_fixture(fail=failed)
                self.assertEqual(result.returncode,7,result.stdout+result.stderr)
                self.assertIn('render',calls)
                self.assertIn('summarize',calls)
                self.assertIn('guard-stop',calls)
                if failed=='asset':
                    self.assertNotIn('prepare',calls)
                    self.assertNotIn('train',calls)
                    self.assertNotIn('predict',calls)

    def test_smoke_uses_separate_training_budget_and_keeps_all_devices(self):
        result,calls,_,cfg,_,second=self.run_fixture(stage='smoke',resume_stage='summarize')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(cfg['sample_steps'],4)
        self.assertEqual(cfg['train_pairs'],48)
        self.assertEqual(cfg['training']['max_steps'],2)
        self.assertEqual(cfg['training']['eval_every'],1)
        self.assertEqual(cfg['training']['monitor_cases'],2)
        self.assertEqual(cfg['cases'],24)
        self.assertIn('fixture_smoke:smoke',result.stdout)
        self.assertIn('fixture_smoke:summarize',second.stdout)
        self.assertEqual(calls.count('train'),1)
        self.assertEqual(calls.count('summarize'),2)

    def test_render_resume_no_teacher_and_no_training(self):
        result,calls,_,cfg,_,second=self.run_fixture(stage='smoke',resume_stage='render')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(calls.count('asset'),2)
        self.assertEqual(calls.count('train'),1)
        self.assertEqual(calls.count('render'),2)
        self.assertEqual(cfg['sample_steps'],4)

    def test_path_overrides_are_single_arguments_and_summary_needs_no_models(self):
        result,calls,_,_,args,_=self.run_fixture(stage='summarize',extra={
            'LEARNED_REFERENCE_LOCAL':'prior outputs with spaces',
            'LEARNED_REFERENCE_PRIMARY':'obs://fixture/original job/anchors',
            'LEARNED_REFERENCE_MIRROR':'obs://fixture/persistent/anchors',
            'LEARNED_BANK_LOCAL':'native bank with spaces',
        })
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],['summarize'])
        self.assertIn('arg=prior outputs with spaces',args)
        self.assertIn('arg=native bank with spaces',args)
        self.assertIn('arg=obs://fixture/original job/anchors',args)

    def test_dry_does_not_spawn_helpers_or_stage_assets(self):
        result,calls,ports,cfg,_,_=self.run_fixture(stage='dry')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(calls,[])
        self.assertEqual(ports,[])
        self.assertIsNone(cfg)
        self.assertIn('trainable RAE decoder residual',result.stdout)


if __name__=='__main__':unittest.main()
