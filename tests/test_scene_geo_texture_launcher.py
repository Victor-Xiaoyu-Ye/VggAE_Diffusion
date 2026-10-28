"""Execute Stage58 Bash orchestration with bounded local cluster-operation stubs."""
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


def argument_groups(text, prefix, entry):
    groups = []
    for line in text.splitlines():
        if not line.startswith(prefix+'='):
            continue
        value = line.split('=', 1)[1]
        if value.endswith('/'+entry):
            groups.append([])
        elif groups:
            groups[-1].append(value)
    return groups


@unittest.skipUnless(BASH, 'Bash required')
class GeoTextureLauncherTests(unittest.TestCase):
    def launch(self, stage='all', staging=0, work=0, merge=0, prepare=0,
               baseline_preflight=0, baseline_train=0, oracle_preflight=0,
               oracle_train=0, publication=0, node=0, arms=None):
        with tempfile.TemporaryDirectory(prefix='geo texture launcher ', dir=REPO.parent) as tmp:
            root = Path(tmp)
            for directory in ('scripts/scale', 'scripts/lib', 'configs'):
                (root/directory).mkdir(parents=True)
            for path in ('scripts/scale/58_train_scene_geo_texture.sh',
                         'scripts/lib/job_cleanup.sh', 'configs/scene_geo_texture_pilot_v1.json'):
                shutil.copyfile(REPO/path, root/path)
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
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
configure_modelarts_distributed() {
 NNODES=3; NUM_NPUS=8; NODE_RANK=${TEST_NODE}; WORLD_SIZE=24; MASTER_ADDR=127.0.0.1
 export NNODES NUM_NPUS NODE_RANK WORLD_SIZE MASTER_ADDR
}
ensure_local_checkpoint() {
 local attempt
 for attempt in {1..100}; do [[ ! -f "${PROJECT}/guard" ]] || break; sleep .02; done
 [[ -f "${PROJECT}/guard" ]] || return 96
 echo staging >> "${PROJECT}/order"
 exit "${TEST_STAGING}"
}
''')
            shell(root/'torchrun-stub.sh', '''#!/bin/bash
set -eu
printf 'torcharg=%s\n' "$@" >> "${PROJECT}/arguments"
while [[ "$1" != -- ]]; do shift; done
shift
printf 'trainarg=%s\n' "$@" >> "${PROJECT}/arguments"
phase=''
while (( $# )); do
 if [[ "$1" == --stage ]]; then phase=$2; shift; fi
 shift
done
echo "${SCENE_GEO_ARM} ${phase}" >> "${PROJECT}/calls"
echo "${SCENE_GEO_ARM}_${phase}" >> "${PROJECT}/order"
if [[ "${SCENE_GEO_ARM}" == baseline ]]; then
 if [[ "${phase}" == preflight ]]; then exit "${TEST_BASELINE_PREFLIGHT}"; else exit "${TEST_BASELINE_TRAIN}"; fi
else
 if [[ "${phase}" == preflight ]]; then exit "${TEST_ORACLE_PREFLIGHT}"; else exit "${TEST_ORACLE_TRAIN}"; fi
fi
''')
            shell(root/'python-stub.sh', '''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  printf 'guardarg=%s\n' "$@" >> "${PROJECT}/arguments"
  trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM
  echo guard-started >> "${PROJECT}/order"; touch "${PROJECT}/guard"
  while true; do sleep .05; done ;;
 */window_run_io.py)
  if [[ "$*" == *--watch* ]]; then
   trap 'echo publisher-stopped >> "${PROJECT}/order"; exit 0' TERM
   while true; do sleep .05; done
  fi
  echo published >> "${PROJECT}/order"; exit "${TEST_PUBLICATION}" ;;
 */coordinated_scene_controls.py)
  echo "worker-coordinator-port=${MASTER_PORT}" >> "${PROJECT}/arguments"
  shift; while [[ "$1" != -- ]]; do shift; done; shift
  "$@" ;;
 */coordinated_diagnostic.py)
  echo "leader-coordinator-port=${MASTER_PORT}" >> "${PROJECT}/arguments"
  shift
  if [[ "${NODE_RANK}" == 0 ]]; then "$@"
  elif [[ "$*" == *"--stage merge"* ]]; then exit "${TEST_MERGE}"
  else exit "${TEST_PREPARE}"; fi ;;
 */audit_scene_geometry_consistency.py)
  printf 'auditarg=%s\n' "$@" >> "${PROJECT}/arguments"
  phase=''
  while (( $# )); do
   if [[ "$1" == --stage ]]; then phase=$2; shift; fi
   shift
  done
  echo "audit_${phase}" >> "${PROJECT}/order"
  if [[ "${phase}" == work ]]; then exit "${TEST_WORK}"; else exit "${TEST_MERGE}"; fi ;;
 */train_scene_geo_texture.py)
  printf 'preparg=%s\n' "$@" >> "${PROJECT}/arguments"
  echo prepare >> "${PROJECT}/order"; exit "${TEST_PREPARE}" ;;
 -) "$REAL_PYTHON" "$@" ;;
 *) echo "Unexpected call: $*" >&2; exit 99 ;;
esac
''')
            env = {k: v for k, v in os.environ.items() if not k.startswith('SCENE_GEO_')}
            env.update(REAL_PYTHON=Path(sys.executable).as_posix(), TEST_NODE=str(node),
                TEST_STAGING=str(staging), TEST_WORK=str(work), TEST_MERGE=str(merge),
                TEST_PREPARE=str(prepare), TEST_BASELINE_PREFLIGHT=str(baseline_preflight),
                TEST_BASELINE_TRAIN=str(baseline_train), TEST_ORACLE_PREFLIGHT=str(oracle_preflight),
                TEST_ORACLE_TRAIN=str(oracle_train), TEST_PUBLICATION=str(publication),
                SCENE_GEO_STAGE=stage, SCENE_GEO_NAMESPACE='fixture', SCENE_GEO_SOURCE_NAMESPACE='source',
                SCENE_NPU_IDLE_GUARD='1', SCENE_HELPER_STOP_TIMEOUT='1', MASTER_PORT='29958')
            env.pop('PYTHONPATH', None)
            if arms is not None:
                env['SCENE_GEO_ARMS'] = arms
            result = subprocess.run([BASH, (root/'scripts/scale/58_train_scene_geo_texture.sh').as_posix()],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=35)
            base = 'fixture_smoke' if stage == 'smoke' else 'fixture'
            logs = root/'output'/base/'launcher'/f'node{node}'
            read = lambda name: (root/name).read_text() if (root/name).is_file() else ''
            records = {p.stem: json.loads(p.read_text()) for p in logs.glob('*.json')} if logs.exists() else {}
            return dict(result=result, order=read('order').splitlines(), calls=read('calls').splitlines(),
                        arguments=read('arguments'), records=records)

    def assert_success(self, result):
        self.assertEqual(result['result'].returncode, 0, result['result'].stdout+result['result'].stderr)

    def test_full_sequence_guard_and_isolated_coordinator_ports(self):
        r = self.launch()
        self.assert_success(r)
        self.assertEqual(r['calls'], ['baseline preflight', 'baseline train', 'oracle_geo preflight', 'oracle_geo train'])
        expected = ['guard-started', 'staging', 'audit_work', 'audit_merge', 'prepare',
                    'baseline_preflight', 'baseline_train', 'oracle_geo_preflight', 'oracle_geo_train', 'guard-stopped']
        positions = [r['order'].index(key) for key in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('guardarg=--devices\nguardarg=8', r['arguments'])
        self.assertEqual(r['records']['effective_config']['dataset']['train_windows'], 512)
        self.assertEqual(r['records']['effective_config']['save_every_steps'], 500)
        self.assertEqual(r['records']['effective_config']['max_steps'], 2000)
        worker_ports = [int(line.split('=')[1]) for line in r['arguments'].splitlines()
                        if line.startswith('worker-coordinator-port=')]
        leader_ports = [int(line.split('=')[1]) for line in r['arguments'].splitlines()
                        if line.startswith('leader-coordinator-port=')]
        torch_ports = [int(line.rsplit('=', 1)[1]) for line in r['arguments'].splitlines()
                       if line.startswith('torcharg=--master_port=')]
        self.assertEqual(worker_ports, [29959, 29962, 29963, 29964, 29965])
        self.assertEqual(leader_ports, [29960, 29961])
        self.assertEqual(torch_ports, [29968, 29969, 29970, 29971])
        self.assertEqual(len(set(worker_ports+leader_ports+torch_ports)), 11)
        self.assertEqual(r['arguments'].count('trainarg=--data-root'), 8)
        self.assertEqual(r['records']['exit']['exit_code'], 0)

    def test_all_real_entrypoint_arguments_and_torchrun_boundary(self):
        r = self.launch()
        self.assert_success(r)
        # Parse captured argv with actual entrypoint and outer torchrun parsers.
        from train_scene_geo_texture import parser as training_parser
        from audit_scene_geometry_consistency import parser as audit_parser
        from torch.distributed.run import get_args_parser
        trains = argument_groups(r['arguments'], 'trainarg', 'train_scene_geo_texture.py')
        prepares = argument_groups(r['arguments'], 'preparg', 'train_scene_geo_texture.py')
        audits = argument_groups(r['arguments'], 'auditarg', 'audit_scene_geometry_consistency.py')
        parsed = [training_parser().parse_args(args) for args in trains+prepares]
        self.assertEqual([args.stage for args in parsed], ['preflight', 'train', 'preflight', 'train', 'prepare'])
        self.assertTrue(all(len(args.data_root) == 2 for args in parsed[:4]))
        checked = [audit_parser().parse_args(args) for args in audits]
        self.assertEqual([args.stage for args in checked], ['work', 'merge'])
        self.assertTrue(all(args.device == 'npu:0' and args.node_rank == 0 and args.nodes == 3 for args in checked))
        values = [line.split('=', 1)[1] for line in r['arguments'].splitlines() if line.startswith('torcharg=')]
        starts = [i for i, v in enumerate(values) if v.startswith('--nnodes=')]
        for start, end in zip(starts, starts[1:]+[len(values)]):
            argv = values[start:end]
            script = next(i for i, v in enumerate(argv) if v.endswith('/train_scene_geo_texture.py'))
            self.assertEqual(argv[script-1], '--')
            outer = get_args_parser().parse_args(argv)
            self.assertEqual(outer.training_script_args, argv[script+1:])
            self.assertEqual(outer.nproc_per_node, '8')

    def test_audit_failure_still_prepares_and_trains_both_arms(self):
        r = self.launch(work=7, merge=9)
        self.assertEqual(r['result'].returncode, 7)
        self.assertIn('audit_merge', r['order'])
        self.assertEqual(len(r['calls']), 4)
        self.assertEqual(r['records']['phase_audit_work']['exit_code'], 7)
        self.assertEqual(r['records']['phase_audit_merge']['exit_code'], 9)

    def test_missing_teacher_does_not_block_cached_training(self):
        r = self.launch(staging=6)
        self.assertEqual(r['result'].returncode, 6)
        self.assertEqual(len(r['calls']), 4)
        self.assertIn('audit_work', r['order'])

    def test_prepare_failure_retains_audit_and_skips_training(self):
        r = self.launch(prepare=8)
        self.assertEqual(r['result'].returncode, 8)
        self.assertIn('audit_merge', r['order'])
        self.assertEqual(r['calls'], [])
        self.assertEqual(r['records']['phase_prepare']['exit_code'], 8)

    def test_failed_preflight_skips_same_arm_training_but_runs_other_arm(self):
        r = self.launch(baseline_preflight=5)
        self.assertEqual(r['result'].returncode, 5)
        self.assertEqual(r['calls'], ['baseline preflight', 'oracle_geo preflight', 'oracle_geo train'])
        self.assertEqual(r['records']['phase_baseline_preflight']['exit_code'], 5)

    def test_failed_training_continues_other_arm(self):
        r = self.launch(baseline_train=4)
        self.assertEqual(r['result'].returncode, 4)
        self.assertEqual(len(r['calls']), 4)
        self.assertEqual(r['records']['phase_oracle_geo_train']['exit_code'], 0)

    def test_smoke_uses_separate_small_config_and_real_resume(self):
        r = self.launch(stage='smoke')
        self.assert_success(r)
        cfg = r['records']['effective_config']
        self.assertEqual(cfg['dataset']['train_windows'], 24)
        self.assertEqual(cfg['dataset']['candidate_windows'], 24)
        self.assertEqual(cfg['dataset']['eval_cases'], 4)
        self.assertEqual(cfg['max_steps'], 3)
        self.assertEqual(cfg['sample_steps'], 4)
        self.assertEqual(cfg['save_every_steps'], 2)
        self.assertIn('fixture_smoke_data', r['arguments'])
        self.assertIn('fixture_smoke_audit', r['arguments'])
        self.assertEqual(len(r['calls']), 4)

    def test_train_only_reuses_preparation_but_does_not_stage_teacher(self):
        r = self.launch(stage='train')
        self.assert_success(r)
        self.assertEqual(r['calls'], ['baseline train', 'oracle_geo train'])
        self.assertIn('prepare', r['order'])
        self.assertNotIn('staging', r['order'])
        self.assertNotIn('audit_work', r['order'])

    def test_audit_only_and_preflight_only(self):
        audit = self.launch(stage='audit')
        self.assert_success(audit)
        self.assertEqual(audit['calls'], [])
        self.assertNotIn('prepare', audit['order'])
        preflight = self.launch(stage='preflight')
        self.assert_success(preflight)
        self.assertEqual(preflight['calls'], ['baseline preflight', 'oracle_geo preflight'])
        self.assertIn('prepare', preflight['order'])
        self.assertNotIn('staging', preflight['order'])

    def test_nonleader_joins_all_phases(self):
        r = self.launch(node=2)
        self.assert_success(r)
        self.assertIn('audit_work', r['order'])
        self.assertNotIn('audit_merge', r['order'])
        self.assertNotIn('prepare', r['order'])
        self.assertEqual(len(r['calls']), 4)
        self.assertEqual(r['records']['phase_prepare']['node'], 2)

    def test_publication_error_and_dry_and_duplicate_arm(self):
        r = self.launch(publication=3)
        self.assertEqual(r['result'].returncode, 73)
        self.assertTrue(r['records']['exit']['publication_failed'])
        dry = self.launch(stage='dry')
        self.assert_success(dry)
        self.assertEqual(dry['order'], [])
        self.assertEqual(dry['records'], {})
        invalid = self.launch(arms='baseline baseline')
        self.assertEqual(invalid['result'].returncode, 2)
        self.assertEqual(invalid['records'], {})


if __name__ == '__main__':
    unittest.main()
