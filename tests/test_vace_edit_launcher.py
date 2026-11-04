"""Run the actual shell entry point with only external jobs replaced by stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from tests.test_scene_matched_launcher import BASH,shell

REPO=Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH,'Bash required')
class EditingLauncherTests(unittest.TestCase):
    def run_fixture(self,fail='',stage='all',aligned=False):
        with tempfile.TemporaryDirectory(prefix='editing launcher ',dir=REPO.parent) as temp:
            root=Path(temp)
            for folder in ('scripts/scale','scripts/lib','configs'): (root/folder).mkdir(parents=True)
            for name in ('scripts/scale/60_train_vace_edit_pilot.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/vace_edit_pilot_v1.json',
                         'scripts/scale/61_train_vace_edit_aligned.sh','configs/vace_edit_aligned_v2.json'):
                shutil.copyfile(REPO/name,root/name)
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
 echo teacher_asset >> "${PROJECT}/calls"
 [[ "${FAIL_PHASE}" != teacher_asset ]] || return 7
}
''')
            shell(root/'torchrun-stub.sh','''#!/bin/bash
while [[ "$1" != -- ]]; do shift; done; shift
"${PROJECT}/python-stub.sh" "$@"
''')
            shell(root/'python-stub.sh','''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  echo guard-start >> "${PROJECT}/calls"
  trap 'echo guard-stop >> "${PROJECT}/calls"; exit 0' TERM
  while true; do sleep .05; done ;;
 */window_run_io.py)
  if [[ "$*" == *--watch* ]]; then
   trap 'exit 0' TERM
   while true; do sleep .05; done
  fi
  echo publish >> "${PROJECT}/calls" ;;
 */coordinated_scene_controls.py)
  echo "port=${MASTER_PORT}" >> "${PROJECT}/ports"
  echo "coordinator $*"
  while [[ "$1" != -- ]]; do shift; done; shift
  "$@" ;;
 -) "$REAL_PYTHON" "$@" ;;
 */check_editing_bank.py)
  key=reuse_bank
  [[ "$*" != *--require-teacher* ]] || key=reuse_teacher
  echo "${key}" >> "${PROJECT}/calls"
  [[ "${FAIL_PHASE}" != "${key}" ]] || exit 7 ;;
 */finalize_vace_edit.py)
  echo finalize >> "${PROJECT}/calls"
  [[ "${FAIL_PHASE}" != finalize ]] || exit 9 ;;
 */prepare_vace_edit.py|*/train_vace_edit.py)
  key=''; stop=''
  while (( $# )); do
   case "$1" in
    --stage|--arm) key=$2;shift ;;
    --stop-after) stop=$2;shift ;;
   esac
   shift
  done
  [[ -z "${stop}" ]] || key="${key}_${stop}"
  echo "${key}" >> "${PROJECT}/calls"
  [[ "${FAIL_PHASE}" != "${key}" ]] || exit 7 ;;
 *) echo "Unexpected call $*"; exit 99 ;;
esac
''')
            # Embedded Python has an isolated path. Inline code imports only
            # atomic(), so the real repository implementation is equivalent here.
            env={k:v for k,v in os.environ.items() if not k.startswith('EDIT_')}
            env.update(REAL_PYTHON=Path(sys.executable).as_posix(),FAIL_PHASE=fail,EDIT_STAGE=stage,
                EDIT_NAMESPACE='fixture',EDIT_ARMS='sft rae',MASTER_PORT='29960',SCENE_HELPER_STOP_TIMEOUT='1')
            if aligned:
                env.pop('EDIT_ARMS');env['EDIT_EVAL_CHECKPOINT']='checkpoints/step_000250.pt'
            entry='61_train_vace_edit_aligned.sh' if aligned else '60_train_vace_edit_pilot.sh'
            result=subprocess.run([BASH,(root/'scripts/scale'/entry).as_posix()],
                env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            calls=(root/'calls').read_text().splitlines()
            ports=(root/'ports').read_text().splitlines()
            ns='fixture_smoke' if stage=='smoke' else 'fixture'
            cfg=json.loads((root/'output'/ns/'launcher/node0/effective_config.json').read_text())
            return result,calls,ports,cfg

    def test_full_order_recovery_and_cleanup(self):
        result,calls,ports,cfg=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        jobs=[x for x in calls if x not in ('guard-start','guard-stop','publish','teacher_asset')]
        self.assertEqual(jobs,['index','cache','native','teacher','sft_2','rae_2','sft_0','rae_0','finalize'])
        self.assertLess(calls.index('guard-start'),calls.index('index'))
        self.assertLess(calls.index('rae_0'),calls.index('guard-stop'))
        self.assertEqual(len(ports),len(set(ports)))

    def test_teacher_failure_preserves_sft(self):
        result,calls,_,_=self.run_fixture('teacher_asset')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('sft_0',calls);self.assertNotIn('rae_2',calls)
        self.assertIn('guard-stop',calls)

    def test_aligned_bank_readonly_all_controls_and_independent_arms(self):
        result,calls,ports,cfg=self.run_fixture(aligned=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        jobs=[x for x in calls if x not in ('guard-start','guard-stop','publish','teacher_asset')]
        self.assertEqual(jobs,['reuse_bank','native','reuse_teacher','sft_2','rae_2','rae_detached_2',
            'region_sft_2','rae_balanced_2','sft_0','rae_0','rae_detached_0','region_sft_0','rae_balanced_0','finalize'])
        self.assertEqual(cfg['eval_cases'],24);self.assertEqual(len(ports),len(set(ports)))
        self.assertIn('--controls',result.stdout)
        self.assertIn('/cache_latents/vace_edit_pilot_v1',result.stdout)

    def test_aligned_teacher_failure_still_runs_both_sft_arms_and_finalizes(self):
        result,calls,_,_=self.run_fixture('reuse_teacher',aligned=True)
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('sft_0',calls);self.assertIn('region_sft_0',calls)
        self.assertNotIn('rae_2',calls);self.assertIn('finalize',calls)

    def test_checkpoint_evaluation_and_sync_never_prepare_or_train(self):
        result,calls,_,_=self.run_fixture(stage='evaluate',aligned=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn('sft_2',calls);self.assertNotIn('cache',calls)
        self.assertIn('--eval-checkpoint checkpoints/step_000250.pt',result.stdout)
        result,calls,_,_=self.run_fixture(stage='sync',aligned=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([x for x in calls if x not in ('guard-start','guard-stop','publish')],['finalize'])

    def test_arm_failure_and_smoke_isolation(self):
        result,calls,_,cfg=self.run_fixture('rae_2','smoke')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('sft_0',calls);self.assertNotIn('rae_0',calls)
        self.assertEqual(cfg['steps'],3);self.assertEqual(cfg['rae_warmup'],0)
        self.assertIn('--join-timeout 3600',result.stdout)
        self.assertIn('--run-id fixture_smoke:smoke',result.stdout)

    def test_index_failure_stops_before_cache_or_training(self):
        result,calls,_,_=self.run_fixture('index','smoke')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertNotIn('cache',calls)
        self.assertNotIn('native',calls)
        self.assertNotIn('sft_2',calls)
        self.assertIn('guard-stop',calls)


if __name__=='__main__': unittest.main()
