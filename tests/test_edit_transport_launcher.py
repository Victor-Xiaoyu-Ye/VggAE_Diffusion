"""Actual Stage62 Bash sequence, with external cluster jobs stubbed."""
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
class TransportLauncherTests(unittest.TestCase):
    def run_fixture(self,stage='all',fail=''):
        with tempfile.TemporaryDirectory(prefix='transport shell ',dir=REPO.parent) as temp:
            root=Path(temp)
            for directory in ('scripts/scale','scripts/lib','configs'):(root/directory).mkdir(parents=True)
            for name in ('scripts/scale/62_audit_edit_transport.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/edit_transport_probe_v1.json'):
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
 echo asset >> "${PROJECT}/calls"
 [[ "${FAIL_PHASE}" != asset ]] || return 7
}
''')
            shell(root/'torchrun-stub.sh','''#!/bin/bash
while [[ "$1" != -- ]]; do shift; done;shift
"${PROJECT}/python-stub.sh" "$@"
''')
            shell(root/'python-stub.sh','''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  echo guard-start >> "${PROJECT}/calls"
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
 */audit_edit_transport.py)
  while [[ "$1" != --phase ]];do shift;done;shift
  echo "$1" >> "${PROJECT}/calls"
  [[ "${FAIL_PHASE}" != "$1" ]] || exit 7 ;;
 *) echo "Unexpected command $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('TRANSPORT_','EDIT_'))}
            env.update(TRANSPORT_STAGE=stage,TRANSPORT_NAMESPACE='fixture',FAIL_PHASE=fail,
                REAL_PYTHON=Path(sys.executable).as_posix(),SCENE_HELPER_STOP_TIMEOUT='1')
            result=subprocess.run([BASH,(root/'scripts/scale/62_audit_edit_transport.sh').as_posix()],
                env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            calls=(root/'calls').read_text().splitlines()
            ports=(root/'ports').read_text().splitlines()
            ns='fixture_smoke' if stage=='smoke' else 'fixture'
            cfg=json.loads((root/'output'/ns/'launcher/node0/effective_config.json').read_text())
            return result,calls,ports,cfg

    def test_independent_order_all_devices_guard_and_persistent_bank(self):
        result,calls,ports,cfg=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],
            ['asset','asset','prepare','rae','vae','summarize'])
        self.assertLess(calls.index('guard-start'),calls.index('prepare'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertEqual(len(ports),len(set(ports)))
        self.assertIn('/cache_latents/vace_edit_pilot_v1',result.stdout)
        self.assertIn('--nproc_per_node=8',result.stdout)
        self.assertNotIn('train_vace_edit.py',result.stdout)

    def test_failed_rae_preserves_vae_and_summary(self):
        result,calls,_,_=self.run_fixture(fail='rae')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('vae',calls);self.assertIn('summarize',calls);self.assertIn('guard-stop',calls)

    def test_smoke_isolation_and_summary_only_does_not_load_models(self):
        result,calls,_,cfg=self.run_fixture(stage='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['iterations'],2);self.assertEqual(cfg['cases'],24)
        self.assertIn('fixture_smoke:smoke',result.stdout)
        result,calls,_,_=self.run_fixture(stage='summarize')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],['summarize'])


if __name__=='__main__':unittest.main()
