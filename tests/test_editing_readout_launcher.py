"""Execute the actual Stage72 shell with bounded cluster process fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_scene_matched_launcher import BASH,shell

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH,'Bash required')
class ReadoutLauncherTests(unittest.TestCase):
    def launch(self,stage='all',fail='',resume=False):
        with tempfile.TemporaryDirectory(prefix='readout launcher ',dir=ROOT.parent) as temp:
            root=Path(temp)
            for folder in ('scripts/scale','scripts/lib','utils','configs'):(root/folder).mkdir(parents=True)
            for name in ('scripts/scale/72_audit_edit_readout_strength.sh','scripts/stage_editing_teacher.sh',
                    'scripts/lib/job_cleanup.sh','configs/editing_readout_strength_v1.json'):
                shutil.copyfile(ROOT/name,root/name)
            (root/'scripts/window_run_io.py').write_text('''import json
from pathlib import Path
def atomic(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
''')
            (root/'utils/__init__.py').write_text('')
            (root/'utils/scene_run.py').write_text('''import json
from pathlib import Path
class ArtifactStore:
 def __init__(self,local,roots):self.local=Path(local)
 def json(self,name,*a):return json.loads((self.local/name).read_text())
''')
            shell(root/'scripts/spatialvid_config.sh','''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output";SCALE_REMOTE_ROOT="${PROJECT}/primary";SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache";VGGAE_REF_ROOT="${PROJECT}/refs";STREAMVGGT_CKPT="${PROJECT}/refs/stream.pt"
PERSISTENT_OBS_ROOT="obs://fixture/owner";PYTHON_BIN="${PROJECT}/python-stub.sh";TORCHRUN_BIN="${PROJECT}/torchrun-stub.sh"
export PROJECT
''')
            shell(root/'scripts/lib/modelarts.sh','''
configure_modelarts_distributed(){
 NNODES=3;NUM_NPUS=8;NODE_RANK=0;WORLD_SIZE=24;MASTER_ADDR=127.0.0.1
 export NNODES NUM_NPUS NODE_RANK WORLD_SIZE MASTER_ADDR
}
ensure_local_checkpoint(){ echo asset >> "${PROJECT}/calls"; }
''')
            shell(root/'torchrun-stub.sh','''#!/bin/bash
echo "torchrun $*" >> "${PROJECT}/args"
while [[ "$1" != -- ]];do shift;done;shift
"${PROJECT}/python-stub.sh" "$@"
''')
            shell(root/'python-stub.sh','''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  echo guard-start >> "${PROJECT}/calls"
  trap 'echo guard-stop >> "${PROJECT}/calls";exit 0' TERM
  while true;do sleep .05;done ;;
 */window_run_io.py)
  if [[ "$*" == *--watch* ]];then trap 'exit 0' TERM;while true;do sleep .05;done;fi
  echo publish >> "${PROJECT}/calls" ;;
 */coordinated_scene_controls.py)
  echo "$MASTER_PORT" >> "${PROJECT}/ports"
  while [[ "$1" != -- ]];do shift;done;shift
  "$@" ;;
 -) "$REAL_PYTHON" "$@" ;;
 */audit_edit_readout_strength.py)
  echo "$*" >> "${PROJECT}/args"
  phase='';arm='';cohort=all;config='';output=''
  while [[ "$#" -gt 0 ]];do
   case "$1" in
    --phase) phase="$2";shift 2;; --arm) arm="$2";shift 2;; --cohort) cohort="$2";shift 2;;
    --config) config="$2";shift 2;; --output) output="$2";shift 2;; *) shift;;
   esac
  done
  call="${phase}:${cohort}${arm:+:${arm}}"
  echo "$call" >> "${PROJECT}/calls"
  echo "$phase cpu=${VGGAE_CPU_ONLY:-unset} autoload=${TORCH_DEVICE_BACKEND_AUTOLOAD:-unset}" >> "${PROJECT}/env"
  if [[ "$phase" == index ]];then
   "$REAL_PYTHON" - "$config" "$output" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
(out/'manifest.json').write_text(json.dumps({'config':json.loads(Path(sys.argv[1]).read_text())}))
PY
  fi
  [[ "$FAIL_PHASE" != "$call" ]] || exit 7 ;;
 *) echo "Unexpected $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('READOUT_','EDIT_')) and k not in ('VGGAE_CPU_ONLY','TORCH_DEVICE_BACKEND_AUTOLOAD')}
            env.update(READOUT_STAGE=stage,READOUT_NAMESPACE='fixture',FAIL_PHASE=fail,
                REAL_PYTHON=Path(sys.executable).as_posix(),MASTER_PORT='31600',SCENE_HELPER_STOP_TIMEOUT='1')
            command=[BASH,(root/'scripts/scale/72_audit_edit_readout_strength.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=ROOT.parent,capture_output=True,text=True,timeout=60)
            ns='fixture_smoke' if stage=='smoke' else 'fixture';path=root/'output'/ns/'launcher/node0/effective_config.json'
            cfg=json.loads(path.read_text()) if path.exists() else None;second=None
            if resume:
                # Fresh local launcher config, restoring from the persisted manifest.
                path.unlink();env.update(READOUT_STAGE='summarize',READOUT_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=ROOT.parent,capture_output=True,text=True,timeout=60)
                self.assertEqual(cfg,json.loads(path.read_text()))
            read=lambda name:(root/name).read_text() if (root/name).exists() else ''
            return result,second,cfg,read('calls').splitlines(),read('env'),read('args'),read('ports').splitlines()

    def test_priority_order_18_render_phases_three_nodes_and_cpu_isolation(self):
        result,_,cfg,calls,env,args,ports=self.launch()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(sum(c.startswith('render:') for c in calls),18)
        self.assertLess(calls.index('summarize:subset'),calls.index('prepare:extension'))
        self.assertIn('render:extension:op_vae',calls);self.assertIn('render:extension:op_rae',calls)
        self.assertEqual(len(ports),len(set(ports)))
        for phase in ('index','evaluate','summarize'):self.assertIn(phase+' cpu=1 autoload=0',env)
        for phase in ('prepare','prepare_vae','render'):self.assertIn(phase+' cpu=unset autoload=unset',env)
        self.assertIn('--nnodes=3 --node_rank=0 --nproc_per_node=8',args)
        self.assertIn('/primary/fixture',args);self.assertIn('/mirror/fixture',args)
        self.assertLess(calls.index('guard-start'),calls.index('prepare:subset'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize:all'))

    def test_independent_failure_continues_and_retains_nonzero_exit(self):
        result,_,_,calls,_,_,_=self.launch(fail='prepare_vae:subset')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('render:subset:op_rae_a055',calls)
        self.assertIn('render:extension:op_rgb',calls);self.assertIn('summarize:all',calls)

    def test_smoke_and_fresh_config_resume(self):
        result,second,cfg,calls,_,_,_=self.launch(stage='smoke',resume=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertTrue(cfg['smoke']);self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(cfg['vae_calibration']['steps'],4)
        self.assertEqual(calls.count('prepare:subset'),1)

    def test_source_contract_failure_stops_before_expensive_work(self):
        result,_,_,calls,_,_,_=self.launch(fail='index:all')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertNotIn('prepare:subset',calls)
        self.assertIn('guard-stop',calls)


if __name__=='__main__':unittest.main()
