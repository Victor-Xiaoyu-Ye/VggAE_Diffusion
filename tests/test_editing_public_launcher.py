"""Execute the actual three-node shell with bounded cluster process stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_scene_matched_launcher import BASH, shell

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH,'Bash required')
class PublicLauncherTests(unittest.TestCase):
    def launch(self,stage='all',fail='',resume=None,fresh=False):
        with tempfile.TemporaryDirectory(prefix='public launcher ',dir=ROOT.parent) as temp:
            root=Path(temp)
            for folder in ('scripts/scale','scripts/lib','utils','configs'): (root/folder).mkdir(parents=True)
            for name in ('scripts/scale/71_audit_public_editing.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/editing_public_colorization_v1.json'):
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
 def __init__(self,local,roots):self.local=Path(local);self.roots=roots
 def json(self,name,required=False):
  for root in [self.local,*self.roots]:
   p=Path(root)/name
   if p.is_file():return json.loads(p.read_text())
  if required:raise FileNotFoundError(name)
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
 */audit_public_editing.py)
  echo "$*" >> "${PROJECT}/args"
  phase='';arm=''
  while [[ "$#" -gt 0 ]];do
   case "$1" in --phase) phase="$2";shift 2;; --arm) arm="$2";shift 2;; *) shift;; esac
  done
  call="${phase}${arm:+:${arm}}"
  echo "$call" >> "${PROJECT}/calls"
  echo "$phase cpu=${VGGAE_CPU_ONLY:-unset} autoload=${TORCH_DEVICE_BACKEND_AUTOLOAD:-unset}" >> "${PROJECT}/env"
  [[ "$FAIL_PHASE" != "$call" ]] || exit 7 ;;
 *) echo "Unexpected $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('PUBLIC_','EDIT_')) and k not in ('VGGAE_CPU_ONLY','TORCH_DEVICE_BACKEND_AUTOLOAD')}
            env.update(PUBLIC_STAGE=stage,PUBLIC_NAMESPACE='fixture',FAIL_PHASE=fail,REAL_PYTHON=Path(sys.executable).as_posix(),
                MASTER_PORT='31500',SCENE_HELPER_STOP_TIMEOUT='1')
            command=[BASH,(root/'scripts/scale/71_audit_public_editing.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=ROOT.parent,capture_output=True,text=True,timeout=60)
            ns='fixture_smoke' if stage=='smoke' else 'fixture'
            config=root/'output'/ns/'launcher/node0/effective_config.json'
            cfg=json.loads(config.read_text()) if config.exists() else None
            second=None
            if resume:
                remote=root/'mirror'/ns; remote.mkdir(parents=True,exist_ok=True)
                (remote/'manifest.json').write_text(json.dumps(dict(config=cfg)))
                if fresh: config.unlink()  # Simulate fresh worker: only durable manifest remains.
                env.update(PUBLIC_STAGE=resume,PUBLIC_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=ROOT.parent,capture_output=True,text=True,timeout=60)
                self.assertEqual(cfg,json.loads(config.read_text()))
            read=lambda name:(root/name).read_text() if (root/name).exists() else ''
            return result,second,cfg,read('calls').splitlines(),read('env'),read('args'),read('ports').splitlines()

    def test_three_nodes_four_arms_cpu_isolation_and_cleanup(self):
        r,_,cfg,calls,env,args,ports=self.launch()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual([c for c in calls if c.startswith('render:')],['render:native','render:sft','render:rgb_full','render:rae_full'])
        self.assertLess(calls.index('render:rgb_full'),calls.index('prepare'))
        self.assertEqual(len(ports),len(set(ports)))
        for phase in ('index','sources','select','shared','weights','evaluate','summarize'):
            self.assertIn(phase+' cpu=1 autoload=0',env)
        for phase in ('text','anchors','cache','prepare','render'):
            self.assertIn(phase+' cpu=unset autoload=unset',env)
        self.assertIn('--nnodes=3 --node_rank=0 --nproc_per_node=8',args)
        self.assertIn('/primary/fixture',args); self.assertIn('/mirror/fixture',args)
        self.assertNotIn('sft',calls)  # No SFT training phase.
        self.assertLess(calls.index('guard-start'),calls.index('index'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))

    def test_failure_still_publishes_all_independent_arms_and_nonzero_exit(self):
        r,_,_,calls,_,_,_=self.launch(fail='prepare')
        self.assertEqual(r.returncode,7,r.stdout+r.stderr)
        for phase in ('render:native','render:sft','render:rgb_full','render:rae_full','evaluate','summarize'):
            self.assertIn(phase,calls)

    def test_smoke_then_fresh_node_resume_keeps_smoke_contract(self):
        r,second,cfg,calls,_,_,_=self.launch(stage='smoke',resume='resume',fresh=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertTrue(cfg['smoke']); self.assertEqual(cfg['sample_steps'],6); self.assertEqual(cfg['cases'],3)
        self.assertEqual(calls.count('prepare'),2)

    def test_index_failure_stops_invalid_contract_before_loading_models(self):
        r,_,_,calls,_,_,_=self.launch(fail='index')
        self.assertEqual(r.returncode,7,r.stdout+r.stderr)
        self.assertNotIn('anchors',calls); self.assertIn('guard-stop',calls)


if __name__=='__main__': unittest.main()
