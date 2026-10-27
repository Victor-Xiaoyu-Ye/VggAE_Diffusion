"""Run the actual stage56 Bash entry with cluster operations stubbed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r'C:\Program Files\Git\bin\bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')


def shell(path, text):
    with path.open('w', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    path.chmod(0o755)


@unittest.skipUnless(BASH, 'Bash required')
class SceneControlLauncherTests(unittest.TestCase):
    def launch(self, stage='all', prep_code=0, train_code=0, arms=None):
        with tempfile.TemporaryDirectory(prefix='control launcher ', dir=REPO.parent) as temporary:
            root = Path(temporary)
            for part in ('scripts/scale', 'scripts/lib', 'configs'):
                (root/part).mkdir(parents=True)
            for part in ('scripts/scale/56_train_scene_control_reference.sh',
                         'scripts/lib/job_cleanup.sh', 'configs/scene_control_reference_v1.json'):
                shutil.copyfile(REPO/part, root/part)
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
''', encoding='utf8')
            shell(root/'scripts/spatialvid_config.sh', '''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output"
SCALE_REMOTE_ROOT="${PROJECT}/primary"
SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache"
STREAMVGGT_CKPT="${PROJECT}/stream.pt"
STREAMVGGT_URL="obs://fixture/stream.pt"
STREAMVGGT_MIRROR_URL="obs://mirror/stream.pt"
PYTHON_BIN="${PROJECT}/python-stub.sh"
TORCHRUN_BIN="${PROJECT}/torchrun-stub.sh"
export PROJECT
''')
            shell(root/'scripts/lib/modelarts.sh', '''
configure_modelarts_distributed() { NNODES=3; NUM_NPUS=8; NODE_RANK=0; WORLD_SIZE=24; MASTER_ADDR=127.0.0.1; }
ensure_local_checkpoint() {
  local attempt
  for attempt in {1..100}; do [[ ! -f "${PROJECT}/guard" ]] || break; sleep .02; done
  [[ -f "${PROJECT}/guard" ]] || return 96
  echo staging >> "${PROJECT}/order"
}
run_torchrun() {
  local stage=''
  printf 'trainarg=%s\n' "$@" >> "${PROJECT}/arguments"
  while (( $# )); do
    if [[ "$1" == --stage ]]; then stage=$2; shift; fi
    shift
  done
  echo "${SCENE_CONTROL_ARM} ${stage}" >> "${PROJECT}/calls"
  echo "train-${stage}" >> "${PROJECT}/order"
  return "${TEST_TRAIN_CODE}"
}
''')
            shell(root/'torchrun-stub.sh', '''#!/bin/bash
set -eu
printf 'workerarg=%s\n' "$@" >> "${PROJECT}/arguments"
echo work >> "${PROJECT}/order"
exit "${TEST_PREP_CODE}"
''')
            shell(root/'python-stub.sh', '''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
  */npu_idle_guard.py)
    trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM
    echo guard-started >> "${PROJECT}/order"
    touch "${PROJECT}/guard"
    while true; do sleep .05; done ;;
  */window_run_io.py)
    if [[ "$*" == *--watch* ]]; then
      trap 'echo publisher-stopped >> "${PROJECT}/order"; exit 0' TERM
      while true; do sleep .05; done
    fi
    echo published >> "${PROJECT}/order" ;;
  */coordinated_diagnostic.py)
    shift
    "$@" ;;
  */coordinated_scene_controls.py)
    shift
    while [[ "$1" != -- ]]; do shift; done
    shift
    "$@" ;;
  */prepare_scene_controls.py)
    printf 'preparg=%s\n' "$@" >> "${PROJECT}/arguments"
    phase=''
    while (( $# )); do
      if [[ "$1" == --stage ]]; then phase=$2; shift; fi
      shift
    done
    echo "${phase}" >> "${PROJECT}/order" ;;
  -)
    "$REAL_PYTHON" "$@" ;;
  *) echo "Unexpected Python call: $*" >&2; exit 99 ;;
esac
''')
            env = dict(os.environ, REAL_PYTHON=Path(sys.executable).as_posix(),
                       TEST_PREP_CODE=str(prep_code), TEST_TRAIN_CODE=str(train_code),
                       SCENE_CONTROL_STAGE=stage, SCENE_CONTROL_NAMESPACE='fixture',
                       SCENE_CONTROL_SOURCE_NAMESPACE='source', SCENE_NPU_IDLE_GUARD='1',
                       SCENE_HELPER_STOP_TIMEOUT='1')
            for key in ('SCENE_CONTROL_CONFIG', 'SCENE_CONTROL_ARMS', 'PYTHONPATH'):
                env.pop(key, None)
            if arms is not None:
                env['SCENE_CONTROL_ARMS']=arms
            result = subprocess.run([BASH, (root/'scripts/scale/56_train_scene_control_reference.sh').as_posix()],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
            base = 'fixture_smoke' if stage=='smoke' else 'fixture'
            logs = root/'output'/base/'launcher/node0'
            read = lambda name: (root/name).read_text() if (root/name).is_file() else ''
            config = logs/'effective_config.json'
            record = logs/'exit.json'
            return dict(result=result,order=read('order'),calls=read('calls'),arguments=read('arguments'),
                        config=json.loads(config.read_text()) if config.is_file() else None,
                        record=json.loads(record.read_text()) if record.is_file() else None)

    def test_full_prepare_then_three_paired_resume_runs(self):
        result=self.launch()
        self.assertEqual(result['result'].returncode,0,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['calls'].splitlines(), [f'{arm} {stage}' for arm in
                         ('baseline','motion','reference') for stage in ('preflight','train')])
        phases=result['order'].splitlines()
        self.assertLess(phases.index('guard-started'),phases.index('staging'))
        self.assertLess(phases.index('plan'),phases.index('work'))
        self.assertLess(phases.index('work'),phases.index('merge'))
        self.assertLess(phases.index('merge'),phases.index('train-preflight'))
        self.assertLess(phases.index('guard-stopped'),phases.index('published'))
        self.assertIn('workerarg=--nproc_per_node=8',result['arguments'])
        self.assertEqual(result['config']['dataset']['train_windows'],4096)
        # Parse every captured real launcher argument list with the actual
        # entrypoint parsers, not a duplicate list of expected flag names.
        from scripts.prepare_scene_controls import parser as prepare_parser
        from train_scene_control import parser as train_parser
        def commands(prefix, entry):
            values=[line.split('=',1)[1] for line in result['arguments'].splitlines()
                    if line.startswith(prefix+'=')]
            groups=[]
            for value in values:
                if value.endswith('/'+entry):
                    groups.append([])
                elif groups:
                    groups[-1].append(value)
            return groups
        prepare_args=commands('preparg','prepare_scene_controls.py')
        prepare_args+=commands('workerarg','prepare_scene_controls.py')
        # Exercise the outer parser too. Python 3.11 eagerly interprets child
        # --local as an ambiguous torchrun abbreviation without this boundary;
        # Python 3.12 accepts it, so parsing alone misses the cluster regression.
        from torch.distributed.run import get_args_parser
        worker_argv=[line.split('=',1)[1] for line in result['arguments'].splitlines()
                     if line.startswith('workerarg=')]
        script_index=next(i for i,value in enumerate(worker_argv)
                          if value.endswith('/prepare_scene_controls.py'))
        self.assertEqual(worker_argv[script_index-1],'--')
        outer=get_args_parser().parse_args(worker_argv)
        self.assertEqual(outer.training_script,worker_argv[script_index])
        self.assertEqual(outer.training_script_args,worker_argv[script_index+1:])
        self.assertIn('--local',outer.training_script_args)
        self.assertEqual(outer.nproc_per_node,'8')
        parsed=[prepare_parser().parse_args(args) for args in prepare_args]
        self.assertEqual({args.stage for args in parsed},{'plan','work','merge'})
        self.assertTrue(all(args.workers==24 and args.device=='npu' and args.train_windows==4096
                            for args in parsed))
        training=[train_parser().parse_args(args) for args in commands('trainarg','train_scene_control.py')]
        self.assertEqual(len(training),6)
        self.assertEqual([args.stage for args in training],['preflight','train']*3)

    def test_smoke_uses_small_separate_config_and_real_resume_path(self):
        result=self.launch('smoke')
        self.assertEqual(result['result'].returncode,0,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['config']['dataset'],dict(train_windows=24,eval_cases=4,source_shard_limit=8))
        self.assertEqual(result['config']['max_steps'],3)
        self.assertEqual(result['config']['sample_steps'],4)
        self.assertIsNone(result['config']['convergence'])
        self.assertIn('fixture_smoke_controls',result['arguments'])
        self.assertEqual(len(result['calls'].splitlines()),6)

    def test_prepare_failure_does_not_start_training_and_preserves_status(self):
        result=self.launch(prep_code=7)
        self.assertEqual(result['result'].returncode,7,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['calls'],'')
        self.assertEqual(result['record']['exit_code'],7)
        self.assertNotIn('merge',result['order'])

    def test_train_only_does_not_stage_teacher_or_reprepare(self):
        result=self.launch('train')
        self.assertEqual(result['result'].returncode,0,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['calls'].splitlines(),[f'{arm} train' for arm in ('baseline','motion','reference')])
        self.assertNotIn('staging',result['order'])
        self.assertNotIn('work\n',result['order'])

    def test_explicit_camera_arm_runs_four_matched_preflight_and_train_pairs(self):
        result=self.launch(arms='baseline camera motion reference')
        self.assertEqual(result['result'].returncode,0,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['calls'].splitlines(),[f'{arm} {stage}' for arm in
            ('baseline','camera','motion','reference') for stage in ('preflight','train')])

    def test_dry_has_no_cluster_mutation(self):
        result=self.launch('dry')
        self.assertEqual(result['result'].returncode,0,result['result'].stdout+result['result'].stderr)
        self.assertEqual(result['order'],'')
        self.assertIsNone(result['record'])


if __name__ == '__main__':
    unittest.main()
