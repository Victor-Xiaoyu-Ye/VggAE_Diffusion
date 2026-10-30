"""Execute Stage59 with local cluster-operation stubs and inspect actual argv."""
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


def arguments(lines, entry):
    groups = []
    for line in lines.splitlines():
        if not line.startswith('runarg='):
            continue
        value = line.split('=', 1)[1]
        if value.endswith('/'+entry):
            groups.append([])
        elif groups:
            groups[-1].append(value)
    return groups


@unittest.skipUnless(BASH, 'Bash required')
class MatchedLauncherTests(unittest.TestCase):
    def launch(self, stage='all', fail='', code=7, publication=0, node=0, arms='dino stream'):
        with tempfile.TemporaryDirectory(prefix='matched launcher ', dir=REPO.parent) as tmp:
            root = Path(tmp)
            for folder in ('scripts/scale', 'scripts/lib', 'configs'):
                (root/folder).mkdir(parents=True)
            for name in ('scripts/scale/59_train_scene_matched_rae.sh',
                         'scripts/lib/job_cleanup.sh', 'configs/scene_matched_rae_v1.json'):
                shutil.copyfile(REPO/name, root/name)
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
 local kind=encoder_assets
 [[ "$3" != *LPIPS* ]] || kind=perceptual_assets
 echo "${kind}" >> "${PROJECT}/order"
 if [[ "${TEST_FAIL}" == "${kind}" ]]; then exit "${TEST_CODE}"; fi
 return 0
}
''')
            shell(root/'torchrun-stub.sh', '''#!/bin/bash
set -eu
printf 'torcharg=%s\n' "$@" >> "${PROJECT}/arguments"
while [[ "$1" != -- ]]; do shift; done
shift
printf 'runarg=%s\n' "$@" >> "${PROJECT}/arguments"
script=$1; shift
arm=''; phase=''; stop=''
while (( $# )); do
 case "$1" in
  --arm) arm=$2; shift ;;
  --stage) phase=$2; shift ;;
  --stop-after) stop=$2; shift ;;
 esac
 shift
done
if [[ "$script" == */prepare_scene_matched.py ]]; then
 key=$phase
else
 key="${arm}_${phase}"
 [[ -z "${stop}" ]] || key="${key}_preflight"
fi
echo "${key}" >> "${PROJECT}/calls"
echo "${key}" >> "${PROJECT}/order"
if [[ "${TEST_FAIL}" == "${key}" ]]; then exit "${TEST_CODE}"; fi
''')
            shell(root/'python-stub.sh', '''#!/bin/bash
set -eu
[[ "$1" != -u ]] || shift
case "$1" in
 */npu_idle_guard.py)
  printf 'guardarg=%s\n' "$@" >> "${PROJECT}/arguments"
  trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM
  echo guard-started >> "${PROJECT}/order"
  while true; do sleep .05; done ;;
 */window_run_io.py)
  if [[ "$*" == *--watch* ]]; then
   trap 'echo publisher-stopped >> "${PROJECT}/order"; exit 0' TERM
   while true; do sleep .05; done
  fi
  echo published >> "${PROJECT}/order"; exit "${TEST_PUBLICATION}" ;;
 */coordinated_scene_controls.py)
  echo "coordinator-port=${MASTER_PORT}" >> "${PROJECT}/arguments"
  shift; while [[ "$1" != -- ]]; do shift; done; shift
  "$@" ;;
 -)
  if [[ "${2:-}" == --check-lpips-assets ]]; then cat >/dev/null; exit 0; fi
  "$REAL_PYTHON" "$@" ;;
 *) echo "Unexpected call: $*" >&2; exit 99 ;;
esac
''')
            env = {k: v for k, v in os.environ.items() if not k.startswith('SCENE_MATCHED_')}
            env.update(REAL_PYTHON=Path(sys.executable).as_posix(), TEST_NODE=str(node),
                TEST_FAIL=fail, TEST_CODE=str(code), TEST_PUBLICATION=str(publication),
                SCENE_MATCHED_STAGE=stage, SCENE_MATCHED_NAMESPACE='fixture',
                SCENE_MATCHED_SOURCE_NAMESPACE='source', SCENE_MATCHED_ARMS=arms,
                SCENE_NPU_IDLE_GUARD='1', SCENE_HELPER_STOP_TIMEOUT='1', MASTER_PORT='29959',
                TORCH_HOME=(root/'torch').as_posix())
            env.pop('PYTHONPATH', None)
            result = subprocess.run([BASH, (root/'scripts/scale/59_train_scene_matched_rae.sh').as_posix()],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=45)
            ns = 'fixture_smoke' if stage == 'smoke' else 'fixture'
            logs = root/'output'/ns/'launcher'/f'node{node}'
            read = lambda name: (root/name).read_text() if (root/name).is_file() else ''
            records = {p.stem: json.loads(p.read_text()) for p in logs.glob('*.json')} if logs.exists() else {}
            return dict(result=result, order=read('order').splitlines(), calls=read('calls').splitlines(),
                        arguments=read('arguments'), records=records)

    def success(self, r):
        self.assertEqual(r['result'].returncode, 0, r['result'].stdout+r['result'].stderr)

    def test_full_sequence_and_distinct_ports(self):
        r = self.launch()
        self.success(r)
        self.assertEqual(r['calls'], ['prepare', 'audit', 'dino_codec_preflight', 'dino_codec',
            'dino_cache', 'dino_flow_preflight', 'dino_flow', 'stream_codec_preflight',
            'stream_codec', 'stream_cache', 'stream_flow_preflight', 'stream_flow'])
        self.assertLess(r['order'].index('guard-started'), r['order'].index('prepare'))
        self.assertLess(r['order'].index('stream_flow'), r['order'].index('guard-stopped'))
        self.assertIn('guardarg=--devices\nguardarg=8', r['arguments'])
        coord = [int(x.split('=')[1]) for x in r['arguments'].splitlines() if x.startswith('coordinator-port=')]
        ports = [int(x.rsplit('=', 1)[1]) for x in r['arguments'].splitlines() if x.startswith('torcharg=--master_port=')]
        self.assertEqual(len(set(coord+ports)), len(coord+ports))
        self.assertEqual(r['records']['exit']['exit_code'], 0)
        self.assertEqual(r['records']['effective_config']['codec']['steps'], 6000)

    def test_audit_failure_does_not_cancel_training(self):
        r = self.launch(fail='audit')
        self.assertEqual(r['result'].returncode, 7)
        self.assertIn('dino_flow', r['calls'])
        self.assertIn('stream_flow', r['calls'])
        self.assertEqual(r['records']['phase_source_audit']['exit_code'], 7)

    def test_arm_failure_continues_independent_arm(self):
        r = self.launch(fail='dino_codec_preflight', code=5)
        self.assertEqual(r['result'].returncode, 5)
        self.assertNotIn('dino_codec', r['calls'])
        self.assertNotIn('dino_cache', r['calls'])
        self.assertIn('stream_flow', r['calls'])

    def test_prepare_failure_skips_dependent_jobs(self):
        r = self.launch(fail='prepare')
        self.assertEqual(r['result'].returncode, 7)
        self.assertEqual(r['calls'], ['prepare'])

    def test_smoke_uses_separate_identity_and_restart(self):
        r = self.launch(stage='smoke')
        self.success(r)
        cfg = r['records']['effective_config']
        self.assertEqual(cfg['codec']['steps'], 3)
        self.assertEqual(cfg['flow_training']['steps'], 3)
        self.assertEqual(cfg['train_windows'], 24)
        self.assertIn('fixture_smoke_dino', r['arguments'])
        self.assertEqual(r['arguments'].count('runarg=--stop-after\nrunarg=2'), 4)

    def test_train_reuses_bank_without_encoder_or_audit(self):
        r = self.launch(stage='train', node=2)
        self.success(r)
        self.assertNotIn('encoder_assets', r['order'])
        self.assertNotIn('prepare', r['calls'])
        self.assertNotIn('audit', r['calls'])
        self.assertEqual(len(r['calls']), 10)
        self.assertIn('torcharg=--node_rank=2', r['arguments'])

    def test_offline_asset_failure_keeps_prepared_audit(self):
        r = self.launch(fail='perceptual_assets')
        self.assertEqual(r['result'].returncode, 7)
        self.assertEqual(r['calls'], ['prepare', 'audit'])

    def test_publication_failure_is_not_success(self):
        r = self.launch(stage='prepare', publication=9)
        self.assertEqual(r['result'].returncode, 73)
        self.assertTrue(r['records']['exit']['publication_failed'])

    def test_actual_entrypoint_parsers_and_torchrun_boundary(self):
        r = self.launch()
        self.success(r)
        from torch.distributed.run import get_args_parser
        values = [x.split('=', 1)[1] for x in r['arguments'].splitlines() if x.startswith('torcharg=')]
        starts = [i for i, x in enumerate(values) if x.startswith('--nnodes=')]
        parsed = []
        for start, end in zip(starts, starts[1:]+[len(values)]):
            argv = values[start:end]
            outer = get_args_parser().parse_args(argv)
            self.assertEqual(outer.nproc_per_node, '8')
            index = argv.index(outer.training_script)
            self.assertEqual(argv[index-1], '--')
            self.assertEqual(outer.training_script_args, argv[index+1:])
            parsed.append(outer)
        self.assertEqual(len(parsed), 12)
        # Root/data agents expose parser() once implementation is complete.
        from prepare_scene_matched import parser as prepare_parser
        from train_scene_matched import parser as train_parser
        for outer in parsed:
            p = prepare_parser() if outer.training_script.endswith('/prepare_scene_matched.py') else train_parser()
            args = p.parse_args(outer.training_script_args)
            self.assertEqual(len(args.root), 2)


if __name__ == '__main__':
    unittest.main()
