"""Execute the actual three-node launcher with bounded cluster process stubs."""
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
class PaperLauncherTests(unittest.TestCase):
    def launch(self,stage='all',fail='',resume=None,extra=None):
        with tempfile.TemporaryDirectory(prefix='paper launcher ',dir=REPO.parent) as temp:
            root=Path(temp)
            for folder in ('scripts/scale','scripts/lib','utils','configs'):(root/folder).mkdir(parents=True)
            for name in ('scripts/scale/69_complete_editing_paper.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/editing_paper_evidence_v1.json'):
                shutil.copyfile(REPO/name,root/name)
            (root/'scripts/window_run_io.py').write_text('''import json
from pathlib import Path
def atomic(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
''')
            (root/'utils/__init__.py').write_text('')
            (root/'utils/scene_run.py').write_text('''class ArtifactStore:
 def __init__(self,*a):pass
 def json(self,*a):return None
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
 */paper_editing_suite.py)
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
            env={k:v for k,v in os.environ.items() if not k.startswith(('PAPER_','EDIT_')) and k not in ('VGGAE_CPU_ONLY','TORCH_DEVICE_BACKEND_AUTOLOAD')}
            env.update(PAPER_STAGE=stage,PAPER_NAMESPACE='fixture',FAIL_PHASE=fail,REAL_PYTHON=Path(sys.executable).as_posix(),
                MASTER_PORT='31000',SCENE_HELPER_STOP_TIMEOUT='1')
            if extra:env.update(extra)
            command=[BASH,(root/'scripts/scale/69_complete_editing_paper.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=60)
            ns='fixture_smoke' if stage=='smoke' else 'fixture'
            cfgpath=root/'output'/ns/'launcher/node0/effective_config.json'
            cfg=json.loads(cfgpath.read_text()) if cfgpath.exists() else None
            second=None
            if resume:
                env.update(PAPER_STAGE=resume,PAPER_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=60)
                self.assertEqual(cfg,json.loads(cfgpath.read_text()))
            read=lambda name:(root/name).read_text() if (root/name).exists() else ''
            return result,second,cfg,read('calls').splitlines(),read('env'),read('args'),read('ports').splitlines()

    def test_all_sequential_three_node_phases_cpu_isolation_and_mirrors(self):
        result,_,cfg,calls,env,args,ports=self.launch()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertLess(calls.index('guard-start'),calls.index('index'))
        self.assertLess(calls.index('sft'),calls.index('render:sft'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertEqual(sum(c.startswith('render:') for c in calls),17)
        self.assertEqual(len(ports),len(set(ports)))
        for phase in ('index','screen','sources','select','weights','evaluate','summarize'):
            self.assertIn(phase+' cpu=1 autoload=0',env)
        for phase in ('prepare','anchors','sft','render'):self.assertIn(phase+' cpu=unset autoload=unset',env)
        self.assertIn('--nnodes=3 --node_rank=0 --nproc_per_node=8',args)
        self.assertIn('/primary/fixture',args);self.assertIn('/mirror/fixture',args)
        self.assertEqual(cfg['cases'],128)

    def test_failed_arm_keeps_independent_arms_evaluation_and_nonzero_exit(self):
        result,_,_,calls,_,_,_=self.launch(fail='render:rae_full')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertIn('render:rae_no_state',calls);self.assertIn('evaluate',calls);self.assertIn('summarize',calls)

    def test_smoke_is_separate_and_resumes_without_retraining(self):
        result,second,cfg,calls,_,_,_=self.launch(stage='smoke',resume='summarize')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(cfg['cases'],3);self.assertTrue(cfg['smoke']);self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(calls.count('sft'),1);self.assertEqual(calls.count('summarize'),2)

    def test_single_arm_resume_and_dry_do_not_train_or_copy_assets(self):
        result,_,_,calls,_,_,_=self.launch(stage='render',extra={'PAPER_ARM':'rae_full'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c.startswith('render:')],['render:rae_full'])
        self.assertNotIn('sft',calls);self.assertNotIn('asset',calls)
        result,_,_,calls,_,_,_=self.launch(stage='dry')
        self.assertEqual(result.returncode,0);self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
