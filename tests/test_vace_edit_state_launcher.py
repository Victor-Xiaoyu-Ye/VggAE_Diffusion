"""Exercise the real Stage65 Bash script with bounded process/cluster stubs."""
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
class VaceEditStateLauncherTests(unittest.TestCase):
    def run_fixture(self,stage='all',fail='',extra=None,resume_stage=None,
                    remote_config=None,local_config=None,explicit_config=None):
        with tempfile.TemporaryDirectory(prefix='vace state shell ',dir=REPO.parent) as temp:
            root=Path(temp)
            for directory in ('scripts/scale','scripts/lib','configs','utils'):
                (root/directory).mkdir(parents=True)
            for name in ('scripts/scale/65_train_vace_edit_state.sh','scripts/stage_editing_teacher.sh',
                         'scripts/lib/job_cleanup.sh','configs/vace_edit_state_v1.json'):
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
 */train_vace_edit_state.py)
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
  echo "$call" >> "${PROJECT}/calls"
  sleep .05
  [[ "${FAIL_PHASE}" != "$call" ]] || exit 7 ;;
 *) echo "Unexpected command $*";exit 99 ;;
esac
''')
            env={k:v for k,v in os.environ.items() if not k.startswith(('STATE_','EDIT_'))}
            env.update(STATE_STAGE=stage,STATE_NAMESPACE='fixture',FAIL_PHASE=fail,
                REAL_PYTHON=Path(sys.executable).as_posix(),SCENE_HELPER_STOP_TIMEOUT='1')
            if extra:env.update(extra)
            ns=env['STATE_NAMESPACE']+('_smoke' if stage=='smoke' else '')
            config_path=root/'output'/ns/'launcher/node0/effective_config.json'
            def configured(kind):
                value=json.loads((root/'configs/vace_edit_state_v1.json').read_text())
                if kind=='smoke':
                    value.update(sample_steps=4,train_pairs=48)
                    value['training'].update(max_steps=2,min_steps=0,eval_every=1,save_every=1,
                        monitor_cases=2,log_every=1,patience=4,preview_every=0)
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
                env['STATE_CONFIG']=explicit.as_posix()
            command=[BASH,(root/'scripts/scale/65_train_vace_edit_state.sh').as_posix()]
            result=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
            cfg=json.loads(config_path.read_text()) if config_path.exists() else None
            second=None
            if resume_stage:
                env.update(STATE_STAGE=resume_stage,STATE_NAMESPACE=ns)
                second=subprocess.run(command,env=env,cwd=REPO.parent,capture_output=True,text=True,timeout=45)
                self.assertEqual(cfg,json.loads(config_path.read_text()),'Resume overwrote effective configuration')
            def lines(name):
                p=root/name
                return p.read_text().splitlines() if p.exists() else []
            return result,lines('calls'),lines('ports'),cfg,lines('arguments'),second

    def test_full_pipeline_dual_roots_and_readonly_sources(self):
        result,calls,ports,cfg,args,_=self.run_fixture()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        phases=[c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases,['asset','asset','states','cache','train:rgb','train:rae',
                                'render:native','render:rgb','render:rae','summarize'])
        self.assertLess(calls.index('guard-start'),calls.index('states'))
        self.assertGreater(calls.index('guard-stop'),calls.index('summarize'))
        self.assertEqual(len(ports),9)
        self.assertEqual(len(set(ports)),9)
        self.assertIn('--devices 8',result.stdout)
        self.assertIn('--nproc_per_node=8',result.stdout)
        self.assertIn('obs://fixture/owner/cache_latents/vace_edit_pilot_v1',result.stdout)
        self.assertTrue(any('/primary/fixture' in a for a in args))
        self.assertTrue(any('/mirror/fixture' in a for a in args))
        self.assertTrue(any('/output/edit_reference_video_v1' in a for a in args))
        self.assertTrue(any('/output/edit_transport_learned_v1' in a for a in args))
        self.assertEqual(args.count('arg=--reference-root'),16)
        self.assertEqual(args.count('arg=--transport-root'),16)
        self.assertEqual(cfg['train_pairs'],0)
        self.assertEqual(cfg['training']['max_steps'],1500)
        self.assertEqual(cfg['state_channels'],dict(rgb=1538,rae=514))

    def test_failure_retains_independent_arms_summary_and_error_exit(self):
        for failed in ('states','cache','train:rgb','render:rgb','asset'):
            with self.subTest(failed=failed):
                result,calls,_,_,_,_=self.run_fixture(fail=failed)
                self.assertEqual(result.returncode,7,result.stdout+result.stderr)
                self.assertIn('train:rae',calls)
                self.assertIn('render:native',calls)
                self.assertIn('render:rae',calls)
                self.assertIn('summarize',calls)
                self.assertIn('guard-stop',calls)
                if failed=='asset':self.assertNotIn('states',calls)

    def test_smoke_isolated_config_and_resumption(self):
        result,calls,_,cfg,_,second=self.run_fixture(stage='smoke',resume_stage='summarize')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(cfg['sample_steps'],4)
        self.assertEqual(cfg['train_pairs'],48)
        self.assertEqual(cfg['training']['max_steps'],2)
        self.assertEqual(cfg['training']['eval_every'],1)
        self.assertEqual(cfg['training']['preview_every'],0)
        self.assertEqual(cfg['training']['monitor_cases'],2)
        self.assertEqual(cfg['cases'],24)
        self.assertIn('fixture_smoke:smoke',result.stdout)
        self.assertIn('fixture_smoke:summarize',second.stdout)
        self.assertEqual(calls.count('train:rgb'),1)
        self.assertEqual(calls.count('train:rae'),1)
        self.assertEqual(calls.count('summarize'),2)

    def test_render_resume_no_teacher_or_training(self):
        result,calls,_,cfg,_,second=self.run_fixture(stage='smoke',resume_stage='render')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual(calls.count('asset'),2)
        self.assertEqual(calls.count('train:rgb'),1)
        self.assertEqual(calls.count('render:rae'),2)
        self.assertEqual(cfg['sample_steps'],4)

    def test_fresh_node_recovers_smoke_config_from_mirror_manifest(self):
        result,calls,_,cfg,_,_=self.run_fixture(stage='render',
            extra={'STATE_NAMESPACE':'fixture_smoke'},remote_config='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],4)
        self.assertEqual(cfg['training']['max_steps'],2)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],
                         ['render:native','render:rgb','render:rae'])

    def test_saved_manifest_and_local_effective_mismatch_is_explicit(self):
        result,calls,_,cfg,_,_=self.run_fixture(stage='render',
            remote_config='smoke',local_config='main')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Local effective config differs from saved manifest',result.stdout+result.stderr)
        self.assertEqual(cfg['training']['max_steps'],1500)
        self.assertNotIn('render:native',calls)
        self.assertIn('guard-stop',calls)

    def test_explicit_config_cannot_override_recovered_run(self):
        result,calls,_,_,_,_=self.run_fixture(stage='render',
            remote_config='smoke',explicit_config='main')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Explicit STATE_CONFIG differs from saved run config',result.stdout+result.stderr)
        self.assertNotIn('render:native',calls)
        self.assertIn('guard-stop',calls)

    def test_matching_explicit_smoke_config_is_accepted_on_fresh_node(self):
        result,_,_,cfg,_,_=self.run_fixture(stage='summarize',
            remote_config='smoke',explicit_config='smoke')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],4)
        self.assertEqual(cfg['training']['max_steps'],2)

    def test_one_arm_training_resume_has_no_teacher_or_other_arm(self):
        result,calls,_,_,args,_=self.run_fixture(stage='train',extra={'STATE_ARMS':'rae'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        phases=[c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases,['train:rae'])
        self.assertIn('arg=rae',args)

    def test_spaced_overrides_are_single_arguments_and_summary_has_no_models(self):
        result,calls,_,_,args,_=self.run_fixture(stage='summarize',extra={
            'STATE_REFERENCE_LOCAL':'prior outputs with spaces',
            'STATE_REFERENCE_PRIMARY':'obs://fixture/original job/anchors',
            'STATE_REFERENCE_MIRROR':'obs://fixture/persistent/anchors',
            'STATE_TRANSPORT_LOCAL':'state cache with spaces',
            'STATE_TRANSPORT_PRIMARY':'obs://fixture/prior transport/cache',
            'STATE_BANK_LOCAL':'native bank with spaces',
        })
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual([c for c in calls if c not in ('publish','guard-start','guard-stop')],['summarize'])
        self.assertIn('arg=prior outputs with spaces',args)
        self.assertIn('arg=native bank with spaces',args)
        self.assertIn('arg=state cache with spaces',args)
        self.assertIn('arg=obs://fixture/original job/anchors',args)
        self.assertIn('arg=obs://fixture/prior transport/cache',args)

    def test_dry_spawns_nothing(self):
        result,calls,ports,cfg,_,_=self.run_fixture(stage='dry')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(calls,[])
        self.assertEqual(ports,[])
        self.assertIsNone(cfg)
        self.assertIn('trainable RGB/RAE state bridge in VACE',result.stdout)


if __name__=='__main__':unittest.main()
