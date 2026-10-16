"""Run the real stage-53 Bash lifecycle with local child/staging stubs.

These tests exercise shell sequencing and cleanup, not torchrun, HCCL or NPU
training. All temporary outputs stay beside the repository on Windows.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r'C:\Program Files\Git\bin\bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')


def write_shell(path, text):
    with path.open('w', encoding='utf8', newline='\n') as handle:
        handle.write(text)
    path.chmod(0o755)


@unittest.skipUnless(BASH, 'Bash is required for launcher lifecycle tests')
class ScenePrefixLauncherTests(unittest.TestCase):
    def run_launcher(self, *, stage='all', preflight_code=0, train_code=0,
                     staging_code=0, publish_code=0, stuck_guard=False):
        temporary_root = REPO.parent if os.name == 'nt' else None
        with tempfile.TemporaryDirectory(prefix='prefix launcher ', dir=temporary_root) as tmp:
            root = Path(tmp)
            for part in ('scripts/scale', 'scripts/lib'):
                (root/part).mkdir(parents=True)
            for part in ('scripts/scale/53_train_scene_prefix_48h.sh',
                         'scripts/lib/job_cleanup.sh'):
                shutil.copyfile(REPO/part, root/part)
            write_shell(root/'scripts/spatialvid_config.sh', '''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output"
SCALE_REMOTE_ROOT="${PROJECT}/primary"
SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache"
STREAMVGGT_CKPT="${PROJECT}/staged/streamvggt.pt"
STREAMVGGT_URL="obs://fixture/encoder"
STREAMVGGT_MIRROR_URL="obs://fixture-mirror/encoder"
PYTHON_BIN="${PROJECT}/python-stub.sh"
export PROJECT
''')
            write_shell(root/'scripts/lib/modelarts.sh', '''
configure_modelarts_distributed() { NNODES=3; NUM_NPUS=8; NODE_RANK=0; }
await_guard() {
  local attempt
  for attempt in {1..200}; do
    [[ ! -f "${PROJECT}/guard-started" ]] || return 0
    sleep 0.02
  done
  echo 'guard stub did not start' >&2
  return 98
}
ensure_local_checkpoint() {
  await_guard
  printf 'checkpoint-staging\\n' >> "${PROJECT}/order"
  printf 'staging-arg=%s\\n' "$@" >> "${PROJECT}/arguments"
  if [[ "${TEST_STAGING_CODE}" != 0 ]]; then
    echo 'checkpoint staging failed'
    return "${TEST_STAGING_CODE}"
  fi
  mkdir -p "$(dirname "$1")"
  printf 'fixture encoder\\n' > "$1"
}
run_torchrun() {
  local stage='' startup='' arg
  printf 'torchrun-call\\n' >> "${PROJECT}/arguments"
  printf 'torchrun-arg=%s\\n' "$@" >> "${PROJECT}/arguments"
  while (( $# )); do
    arg=$1; shift
    case "${arg}" in
      --stage) stage=$1; shift ;;
      --startup-seconds) startup=$1; shift ;;
    esac
  done
  [[ "${startup}" =~ ^[0-9]+$ ]] || { echo 'missing/invalid startup-seconds'; return 97; }
  printf '%s %s\\n' "${stage}" "${startup}" >> "${PROJECT}/calls"
  printf 'run-%s\\n' "${stage}" >> "${PROJECT}/order"
  echo "last ${stage} output"
  case "${stage}" in
    preflight) return "${TEST_PREFLIGHT_CODE}" ;;
    train) return "${TEST_TRAIN_CODE}" ;;
    *) return 96 ;;
  esac
}
''')
            write_shell(root/'python-stub.sh', '''#!/bin/bash
set -eu
case "$1" in
  */npu_idle_guard.py)
    printf 'guard-arg=%s\\n' "$@" >> "${PROJECT}/guard-arguments"
    if [[ "${TEST_STUCK_GUARD}" == 1 ]]; then
      trap '' TERM
    else
      trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM
    fi
    echo guard-started >> "${PROJECT}/order"
    touch "${PROJECT}/guard-started"
    while true; do sleep 0.05; done
    ;;
  */window_run_io.py)
    if [[ "$*" == *--watch* ]]; then
      trap 'echo publisher-stopped >> "${PROJECT}/order"; exit 0' TERM
      while true; do sleep 0.05; done
    fi
    echo final-publish >> "${PROJECT}/order"
    printf 'publish-arg=%s\\n' "$@" >> "${PROJECT}/publish-arguments"
    if [[ "${TEST_PUBLISH_CODE}" != 0 ]]; then
      echo "stub final publication failed: ${TEST_PUBLISH_CODE}" >&2
    fi
    exit "${TEST_PUBLISH_CODE}"
    ;;
  -)
    cat >/dev/null
    failed=false
    [[ "$4" != failed ]] || failed=true
    printf '{"exit_code":%s,"publication_failed":%s}\\n' "$3" "${failed}" > "$2/exit.json"
    ;;
  *) echo "unexpected Python stub invocation: $1" >&2; exit 99 ;;
esac
''')
            env = dict(os.environ, SCENE_HELPER_STOP_TIMEOUT='1',
                SCENE_SOURCE_NAMESPACE='source', SCENE_PREFIX_NAMESPACE='prefix',
                SCENE_PREFIX_STAGE=stage, SCENE_NPU_IDLE_GUARD='1',
                TEST_STUCK_GUARD=str(int(stuck_guard)), TEST_PREFLIGHT_CODE=str(preflight_code),
                TEST_TRAIN_CODE=str(train_code), TEST_STAGING_CODE=str(staging_code),
                TEST_PUBLISH_CODE=str(publish_code))
            started = time.monotonic()
            result = subprocess.run([BASH, (root/'scripts/scale/53_train_scene_prefix_48h.sh').as_posix()],
                env=env, cwd=root, capture_output=True, text=True, timeout=20)
            elapsed = time.monotonic()-started
            logs = root/'output/prefix/launcher/node0'
            self.assertTrue((logs/'exit.json').is_file(), result.stdout+result.stderr)
            calls = (root/'calls').read_text().splitlines() if (root/'calls').exists() else []
            return dict(result=result, elapsed=elapsed,
                record=json.loads((logs/'exit.json').read_text()),
                log=(logs/'pipeline.log').read_text(), order=(root/'order').read_text(),
                calls=[line.split() for line in calls], arguments=(root/'arguments').read_text(),
                publish_arguments=(root/'publish-arguments').read_text())

    def test_preflight_then_fresh_train_with_startup_accounting_and_complete_tail(self):
        run = self.run_launcher()
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertEqual(run['record'], dict(exit_code=0, publication_failed=False))
        self.assertEqual([stage for stage, _ in run['calls']], ['preflight', 'train'])
        self.assertTrue(all(int(seconds) >= 0 for _, seconds in run['calls']))
        self.assertEqual(run['arguments'].count('torchrun-arg=--startup-seconds\n'), 2)
        self.assertEqual(run['arguments'].count('torchrun-call\n'), 2)
        self.assertLess(run['order'].index('checkpoint-staging'), run['order'].index('run-preflight'))
        self.assertLess(run['order'].index('run-preflight'), run['order'].index('run-train'))
        self.assertLess(run['order'].index('guard-stopped'), run['order'].index('final-publish'))
        self.assertLess(run['order'].index('publisher-stopped'), run['order'].index('final-publish'))
        for line in ('last preflight output', 'last train output', 'Scene prefix run complete:',
                     'Scene prefix launcher finished: node=0, exit_code=0'):
            self.assertIn(line, run['log'])
        self.assertEqual(run['publish_arguments'].count('publish-arg=--root\n'), 2)
        self.assertIn('/primary/prefix/launcher/node0', run['publish_arguments'])
        self.assertIn('/mirror/prefix/launcher/node0', run['publish_arguments'])

    def test_preflight_failure_prevents_training_and_preserves_original_exit(self):
        run = self.run_launcher(preflight_code=7, publish_code=9)
        self.assertEqual(run['result'].returncode, 7, run['result'].stdout+run['result'].stderr)
        self.assertEqual(run['record'], dict(exit_code=7, publication_failed=True))
        self.assertEqual([stage for stage, _ in run['calls']], ['preflight'])
        self.assertNotIn('run-train', run['order'])
        self.assertNotIn('Scene prefix run complete:', run['log'])
        self.assertIn('last preflight output', run['log'])
        self.assertIn('Scene prefix launcher finished: node=0, exit_code=7', run['log'])

    def test_staging_failure_prevents_both_stages_and_still_cleans_helpers(self):
        run = self.run_launcher(staging_code=11)
        self.assertEqual(run['result'].returncode, 11, run['result'].stdout+run['result'].stderr)
        self.assertEqual(run['calls'], [])
        self.assertEqual(run['record']['exit_code'], 11)
        self.assertIn('checkpoint staging failed', run['log'])
        self.assertLess(run['order'].index('guard-stopped'), run['order'].index('final-publish'))

    def test_guard_ignoring_term_is_killed_with_bounded_cleanup(self):
        run = self.run_launcher(stuck_guard=True)
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertLess(run['elapsed'], 15)
        self.assertIn('NPU idle guard exceeded 1s; sending SIGKILL', run['log'])
        self.assertIn('Scene prefix launcher finished:', run['log'])
        self.assertIn('final-publish', run['order'])

    def test_final_publication_failure_is_not_reported_as_success(self):
        run = self.run_launcher(publish_code=9)
        self.assertEqual(run['result'].returncode, 73, run['result'].stdout+run['result'].stderr)
        self.assertEqual(run['record'], dict(exit_code=73, publication_failed=True))
        self.assertEqual(run['order'].count('final-publish'), 2)
        self.assertIn('last train output', run['log'])
        self.assertIn('stub final publication failed: 9', run['result'].stderr)

    def test_explicit_train_only_does_not_repeat_preflight(self):
        run = self.run_launcher(stage='train', train_code=17)
        self.assertEqual(run['result'].returncode, 17, run['result'].stdout+run['result'].stderr)
        self.assertEqual([stage for stage, _ in run['calls']], ['train'])
        self.assertEqual(run['record']['exit_code'], 17)
        self.assertNotIn('run-preflight', run['order'])
        self.assertIn('last train output', run['log'])

    def test_real_training_parser_exposes_startup_seconds(self):
        # Run only argparse help: no distributed initialization, model loading
        # or training. This catches launcher/parser option drift.
        result = subprocess.run([sys.executable, str(REPO/'train_scene_prefix.py'), '--help'],
            cwd=REPO, env=os.environ, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('--startup-seconds STARTUP_SECONDS', result.stdout)
        self.assertIn('--stage {preflight,train}', result.stdout)


if __name__ == '__main__':
    unittest.main()
