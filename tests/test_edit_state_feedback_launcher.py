"""Exercise the real Stage66 Bash script with bounded process/cluster stubs."""
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
class EditStateFeedbackLauncherTests(unittest.TestCase):
    def run_fixture(self,stage='all',fail='',extra=None,resume_stage=None,
                    remote_config=None,local_config=None,explicit_config=None):
        with tempfile.TemporaryDirectory(prefix='edit feedback shell ',dir=REPO.parent) as temp:
            root=Path(temp)
            for directory in ('scripts/scale','scripts/lib','configs','utils'):
                (root/directory).mkdir(parents=True)
            for name in ('scripts/scale/66_refine_vace_edit_state.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/edit_state_feedback_v1.json'):
                shutil.copyfile(REPO/name,root/name)
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
''',encoding='utf8')
            (root/'utils/__init__.py').write_text('',encoding='utf8')
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
  call="${phase}${arm:+:${arm}}"
  if [[ "$phase" == summarize ]];then
   echo "summary diagnostics: fault=${PYTHONFAULTHANDLER:-} unbuffered=${PYTHONUNBUFFERED:-}"
  fi
  echo "$call" >> "${PROJECT}/calls"
  sleep .05
  [[ "${FAIL_PHASE}" != "$call" ]] || exit 7 ;;
 *) echo "Unexpected command $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('FEEDBACK_','EDIT_'))}
            env.update(FEEDBACK_STAGE=stage,FEEDBACK_NAMESPACE='fixture',FAIL_PHASE=fail,
                REAL_PYTHON=Path(sys.executable).as_posix(),SCENE_HELPER_STOP_TIMEOUT='1',MASTER_PORT='30700')
            if extra:env.update(extra)
            ns=env['FEEDBACK_NAMESPACE']+('_smoke' if stage=='smoke' else '')
            config_path=root/'output'/ns/'launcher/node0/effective_config.json'
            def configured(kind):
                value=json.loads((root/'configs/edit_state_feedback_v1.json').read_text())
                if kind=='smoke':
                    value.update(cases=3,sample_steps=6)
                    value['solver'].update(sweeps=1,steps=1)
                return value
            if remote_config:
                remote=root/'mirror'/ns/'manifest.json'
                remote.parent.mkdir(parents=True,exist_ok=True)
                remote.write_text(json.dumps(dict(config=configured(remote_config))),encoding='utf8')
            if local_config:
                config_path.parent.mkdir(parents=True,exist_ok=True)
                config_path.write_text(json.dumps(configured(local_config)),encoding='utf8')
            if explicit_config:
                explicit=root/'explicit config.json'
                explicit.write_text(json.dumps(configured(explicit_config)),encoding='utf8')
                env['FEEDBACK_CONFIG']=explicit.as_posix()
            command=[BASH,(root/'scripts/scale/66_refine_vace_edit_state.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            cfg=json.loads(config_path.read_text()) if config_path.exists() else None
            second=None
            if resume_stage:
                env.update(FEEDBACK_STAGE=resume_stage,FEEDBACK_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
                self.assertEqual(cfg,json.loads(config_path.read_text()),'Resume overwrote effective configuration')
            def lines(name):
                p=root/name
                return p.read_text().splitlines() if p.exists() else []
            return result,lines('calls'),lines('ports'),cfg,lines('arguments'),second

    def test_full_pipeline_dual_roots_and_readonly_sources(self):
        result,calls,ports,cfg,args,_=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('summary diagnostics: fault=1 unbuffered=1',result.stdout)
        phases=[c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases,['asset','asset','prepare','render:native','render:rgb_base',
                                'render:rae_base','render:rgb_joint','render:rae_joint','summarize'])
        self.assertLess(calls.index('guard-start'),calls.index('prepare'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertEqual(len(ports),8)
        self.assertEqual(len(set(ports)),8)
        self.assertIn('--devices 8',result.stdout)
        self.assertIn('--nproc_per_node=8',result.stdout)
        self.assertTrue(any('/primary/fixture' in a for a in args))
        self.assertTrue(any('/mirror/fixture' in a for a in args))
        self.assertTrue(any('/output/vace_edit_state_v1' in a for a in args))
        self.assertTrue(any('/output/edit_transport_learned_v1' in a for a in args))
        self.assertEqual(args.count('arg=--state-root'),14)
        self.assertEqual(args.count('arg=--transport-root'),14)
        self.assertEqual(cfg['cases'],24)
        self.assertEqual(cfg['sample_steps'],30)
        self.assertNotIn('train',phases)

    def test_failure_retains_independent_arms_and_error_exit(self):
        for failed in ('prepare','render:rgb_joint','render:rae_base','asset'):
            with self.subTest(failed=failed):
                result,calls,_,_,_,_=self.run_fixture(fail=failed)
                self.assertEqual(result.returncode,7,result.stdout+result.stderr)
                for arm in ('native','rgb_base','rae_base','rgb_joint','rae_joint'):
                    self.assertIn('render:'+arm,calls)
                self.assertIn('summarize',calls)
                self.assertIn('guard-stop',calls)
                if failed=='asset':self.assertNotIn('prepare',calls)

    def test_smoke_isolated_and_uses_full_stage65(self):
        result,calls,_,cfg,args,second=self.run_fixture(stage='smoke',resume_stage='summarize')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(cfg['cases'],3)
        self.assertEqual(cfg['solver']['sweeps'],1)
        self.assertEqual(cfg['solver']['steps'],1)
        self.assertIn('fixture_smoke:smoke',result.stdout)
        self.assertIn('fixture_smoke:summarize',second.stdout)
        self.assertEqual(calls.count('prepare'),1)
        self.assertEqual(calls.count('summarize'),2)
        self.assertTrue(any('/output/vace_edit_state_v1' in a for a in args))
        self.assertFalse(any('vace_edit_state_v1_smoke' in a for a in args))

    def test_render_resume_no_teacher_or_prepare(self):
        result,calls,_,cfg,_,second=self.run_fixture(stage='smoke',resume_stage='render')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(calls.count('asset'),2)
        self.assertEqual(calls.count('prepare'),1)
        self.assertEqual(calls.count('render:rae_joint'),2)
        self.assertEqual(cfg['sample_steps'],6)

    def test_fresh_node_recovers_smoke_config_from_mirror(self):
        result,calls,_,cfg,_,_=self.run_fixture(stage='render',
            extra={'FEEDBACK_NAMESPACE':'fixture_smoke'},remote_config='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(cfg['cases'],3)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],
                         ['render:native','render:rgb_base','render:rae_base','render:rgb_joint','render:rae_joint'])

    def test_saved_manifest_and_local_effective_mismatch_is_explicit(self):
        result,calls,_,cfg,_,_=self.run_fixture(stage='render',
            remote_config='smoke',local_config='main')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Local effective config differs from saved manifest',result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],30)
        self.assertNotIn('render:native',calls)
        self.assertIn('guard-stop',calls)

    def test_explicit_config_cannot_override_recovered_run(self):
        result,calls,_,_,_,_=self.run_fixture(stage='render',
            remote_config='smoke',explicit_config='main')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Explicit FEEDBACK_CONFIG differs from saved run config',result.stdout+result.stderr)
        self.assertNotIn('render:native',calls)

    def test_matching_explicit_smoke_config_is_accepted(self):
        result,_,_,cfg,_,_=self.run_fixture(stage='summarize',
            remote_config='smoke',explicit_config='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],6)
        self.assertEqual(cfg['solver']['steps'],1)

    def test_arm_resume_keeps_stable_ports_and_no_prepare(self):
        result,calls,ports,_,args,_=self.run_fixture(stage='render',
            extra={'FEEDBACK_ARMS':'rae_joint rgb_base'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],
                         ['render:rae_joint','render:rgb_base'])
        self.assertEqual(ports,['30724','30721'])
        self.assertIn('arg=rae_joint',args)

    def test_spaced_overrides_are_single_arguments(self):
        result,calls,_,_,args,_=self.run_fixture(stage='summarize',extra={
            'FEEDBACK_STATE_LOCAL':'prior states with spaces',
            'FEEDBACK_STATE_PRIMARY':'obs://fixture/original job/states',
            'FEEDBACK_STATE_MIRROR':'obs://fixture/persistent/states',
            'FEEDBACK_TRANSPORT_LOCAL':'state cache with spaces',
            'FEEDBACK_TRANSPORT_PRIMARY':'obs://fixture/prior transport/cache',
        })
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],['summarize'])
        self.assertIn('arg=prior states with spaces',args)
        self.assertIn('arg=state cache with spaces',args)
        self.assertIn('arg=obs://fixture/original job/states',args)
        self.assertIn('arg=obs://fixture/prior transport/cache',args)

    def test_invalid_or_duplicate_arms_fail_before_work(self):
        for arms in ('rae','rae_joint rae_joint','rgb_joint wrong'):
            result,calls,ports,cfg,_,_=self.run_fixture(extra={'FEEDBACK_ARMS':arms})
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertEqual(calls,[])
            self.assertEqual(ports,[])
            self.assertIsNone(cfg)

    def test_dry_spawns_nothing(self):
        result,calls,ports,cfg,_,_=self.run_fixture(stage='dry')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(calls,[])
        self.assertEqual(ports,[])
        self.assertIsNone(cfg)
        self.assertIn('Frozen selected RGB/RAE models',result.stdout)


if __name__=='__main__':unittest.main()
