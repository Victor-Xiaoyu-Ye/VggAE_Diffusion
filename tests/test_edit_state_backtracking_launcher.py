"""Real Bash Stage67 wrapper + Stage66 delegation with bounded cluster stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_scene_matched_launcher import BASH, shell

REPO = Path(__file__).resolve().parents[1]
BACKTRACK = dict(optimizer='adaptive_backtracking', backtrack_max_trials=8,
    backtrack_shrink=.5, step_grow=1.5, min_lr=1e-6, max_lr=.03, armijo=1e-4)


@unittest.skipUnless(BASH, 'Bash required')
class EditStateBacktrackingLauncherTests(unittest.TestCase):
    def run_fixture(self, stage='all', extra=None, resume=None, remote=None, explicit=None,
                    entry='67_refine_vace_edit_state_backtracking.sh',
                    config_name='edit_state_feedback_bt_v2.json', namespace_default='edit_state_feedback_bt_v2'):
        with tempfile.TemporaryDirectory(prefix='edit backtracking shell ', dir=REPO.parent) as temp:
            root = Path(temp)
            for directory in ('scripts/scale','scripts/lib','configs','utils'):
                (root/directory).mkdir(parents=True)
            for name in ('scripts/scale/66_refine_vace_edit_state.sh',
                    'scripts/scale/67_refine_vace_edit_state_backtracking.sh',
                    'scripts/scale/68_refine_vace_edit_state_context.sh',
                    'scripts/stage_editing_teacher.sh','scripts/lib/job_cleanup.sh',
                    'configs/edit_state_feedback_v1.json','configs/edit_state_feedback_bt_v2.json',
                    'configs/edit_state_context_sync_v1.json'):
                shutil.copyfile(REPO/name, root/name)
            (root/'utils/__init__.py').write_text('')
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
''', encoding='utf8')
            (root/'utils/scene_run.py').write_text('''
import json
from pathlib import Path
class ArtifactStore:
    def __init__(self,local,roots):self.local=Path(local);self.roots=roots
    def json(self,name,required=False):
        for root in [self.local,*map(Path,self.roots)]:
            path=root/name
            if path.exists():return json.loads(path.read_text())
        if required:raise FileNotFoundError(name)
        return None
''', encoding='utf8')
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
ensure_local_checkpoint() { echo asset >> "${PROJECT}/calls"; }
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
 */refine_vace_edit_state.py)
  printf 'arg=%s\n' "$@" >> "${PROJECT}/arguments"
  phase='';arm=''
  while [[ "$#" -gt 0 ]];do
   case "$1" in
    --phase) phase="$2";shift 2 ;;
    --arm) arm="$2";shift 2 ;;
    *) shift ;;
   esac
  done
  echo "phase environment: ${phase} cpu=${VGGAE_CPU_ONLY:-unset} autoload=${TORCH_DEVICE_BACKEND_AUTOLOAD:-unset}"
  echo "${phase}${arm:+:${arm}}" >> "${PROJECT}/calls"
  [[ -z "${STUB_FAIL_ARM:-}" || "${arm}" != "${STUB_FAIL_ARM}" ]] || exit 17
  sleep .05 ;;
 *) echo "Unexpected command $*";exit 99 ;;
esac
''')
            env = {k:v for k,v in os.environ.items() if not k.startswith(('FEEDBACK_','EDIT_'))
                and k not in ('MASTER_PORT','DEFAULT_FEEDBACK_CONFIG','DEFAULT_FEEDBACK_ARMS','STUB_FAIL_ARM',
                              'VGGAE_CPU_ONLY','TORCH_DEVICE_BACKEND_AUTOLOAD')}
            env.update(FEEDBACK_STAGE=stage, REAL_PYTHON=Path(sys.executable).as_posix(),
                SCENE_HELPER_STOP_TIMEOUT='1')
            if extra:
                env.update(extra)
            namespace = env.get('FEEDBACK_NAMESPACE',namespace_default)+('_smoke' if stage=='smoke' else '')
            config_path = root/'output'/namespace/'launcher/node0/effective_config.json'
            main = json.loads((root/'configs'/config_name).read_text())
            def config(kind):
                value = json.loads(json.dumps(main))
                if kind == 'smoke':
                    value.update(cases=3,sample_steps=6)
                    value['solver'].update(sweeps=1,steps=1)
                elif kind == 'custom':
                    value['sample_steps']=4
                    value['solver']['lr']=.015
                return value
            if remote:
                manifest = root/'mirror'/namespace/'manifest.json'
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps(dict(config=config(remote))),encoding='utf8')
            if explicit:
                path = root/'custom configuration with spaces.json'
                path.write_text(json.dumps(config(explicit)),encoding='utf8')
                env['FEEDBACK_CONFIG'] = path.as_posix()
            command = [BASH,(root/'scripts/scale'/entry).as_posix()]
            result = subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            effective = json.loads(config_path.read_text()) if config_path.exists() else None
            second = None
            if resume:
                env.update(FEEDBACK_STAGE=resume,FEEDBACK_NAMESPACE=namespace)
                second = subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
                self.assertEqual(effective,json.loads(config_path.read_text()))
            def lines(name):
                path=root/name
                return path.read_text().splitlines() if path.exists() else []
            return result,lines('calls'),lines('ports'),effective,lines('arguments'),second

    def test_config_changes_only_optimizer(self):
        original=json.loads((REPO/'configs/edit_state_feedback_v1.json').read_text())
        candidate=json.loads((REPO/'configs/edit_state_feedback_bt_v2.json').read_text())
        for name,value in BACKTRACK.items():
            self.assertEqual(candidate['solver'].pop(name),value)
        self.assertEqual(candidate,original)

    def test_default_full_delegation_config_ports_and_lifecycle(self):
        result,calls,ports,cfg,args,_=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('phase environment: summarize cpu=1 autoload=0',result.stdout)
        self.assertIn('phase environment: render cpu=unset autoload=unset',result.stdout)
        phases=[c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases,['asset','asset','prepare','render:native','render:rgb_base',
            'render:rae_base','render:rgb_joint','render:rae_joint','summarize'])
        self.assertEqual(ports,['30802','30803','30820','30821','30822','30823','30824','30830'])
        self.assertEqual(cfg,json.loads((REPO/'configs/edit_state_feedback_bt_v2.json').read_text()))
        self.assertLess(calls.index('guard-start'),calls.index('prepare'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertIn('--nproc_per_node=8',result.stdout)
        self.assertIn('--nnodes=3',result.stdout)
        self.assertIn('--master_port=30883',result.stdout)
        self.assertTrue(any('/primary/edit_state_feedback_bt_v2' in a for a in args))
        self.assertTrue(any('/mirror/edit_state_feedback_bt_v2' in a for a in args))
        self.assertTrue(any('/output/vace_edit_state_v1' in a for a in args))
        self.assertTrue(any('/output/edit_transport_learned_v1' in a for a in args))
        self.assertNotIn('train',phases)

    def test_smoke_uses_full_checkpoint_and_resumes_without_explicit_default(self):
        result,calls,_,cfg,args,second=self.run_fixture(stage='smoke',resume='render')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual((cfg['cases'],cfg['sample_steps'],cfg['solver']['sweeps'],cfg['solver']['steps']),(3,6,1,1))
        self.assertEqual(cfg['solver']['optimizer'],'adaptive_backtracking')
        self.assertTrue(any('/output/edit_state_feedback_bt_v2_smoke' in a for a in args))
        self.assertTrue(any('/output/vace_edit_state_v1' in a for a in args))
        self.assertFalse(any('vace_edit_state_v1_smoke' in a for a in args))
        self.assertEqual(calls.count('prepare'),1)
        self.assertEqual(calls.count('render:rae_joint'),2)

    def test_fresh_node_restores_smoke_from_mirror(self):
        result,calls,ports,cfg,_,_=self.run_fixture(stage='render',remote='smoke',
            extra={'FEEDBACK_NAMESPACE':'edit_state_feedback_bt_v2_smoke','FEEDBACK_ARMS':'rae_joint'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(cfg['solver']['optimizer'],'adaptive_backtracking')
        self.assertEqual(ports,['30824'])
        self.assertNotIn('asset',calls)
        self.assertNotIn('prepare',calls)

    def test_user_config_namespace_and_port_override(self):
        result,_,ports,cfg,_,_=self.run_fixture(stage='render',explicit='custom',
            extra={'FEEDBACK_NAMESPACE':'custom_bt','MASTER_PORT':'31200','FEEDBACK_ARMS':'rgb_joint rae_joint'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(ports,['31223','31224'])
        self.assertEqual(cfg['sample_steps'],4)
        self.assertEqual(cfg['solver']['lr'],.015)
        self.assertIn('custom configuration with spaces.json',result.stdout)

    def test_explicit_override_cannot_replace_resumed_config(self):
        result,calls,_,_,_,_=self.run_fixture(stage='render',remote='smoke',explicit='main')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Explicit FEEDBACK_CONFIG differs from saved run config',result.stdout+result.stderr)
        self.assertNotIn('render:native',calls)

    def test_matching_explicit_smoke_override_is_accepted(self):
        result,_,ports,cfg,_,_=self.run_fixture(stage='summarize',remote='smoke',explicit='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(ports,['30830'])
        self.assertEqual(cfg['sample_steps'],6)

    def test_dry_launches_no_work(self):
        result,calls,ports,cfg,_,_=self.run_fixture(stage='dry')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(calls,[])
        self.assertEqual(ports,[])
        self.assertIsNone(cfg)
        self.assertIn('edit_state_feedback_bt_v2',result.stdout)


if __name__=='__main__':
    unittest.main()
