"""Run the real replay launcher with shell-only assets and child-process stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') or (
    r'C:\Program Files\Git\bin\bash.exe'
    if Path(r'C:\Program Files\Git\bin\bash.exe').is_file() else None)


@unittest.skipUnless(BASH, 'Bash is required for launcher lifecycle tests')
class SceneLauncherCleanupTests(unittest.TestCase):
    def run_launcher(self, *, stuck_guard=False, task_code=0, publish_code=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ('scripts/scale', 'scripts/lib'):
                (root/part).mkdir(parents=True)
            for part in ('scripts/scale/52_audit_scene_flow_weights.sh',
                         'scripts/lib/job_cleanup.sh'):
                shutil.copyfile(REPO/part, root/part)
            (root/'scripts/spatialvid_config.sh').write_text('''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output"
SCALE_REMOTE_ROOT="${PROJECT}/primary"
SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache"
PYTHON_BIN="${PROJECT}/python-stub.sh"
export PROJECT
''', encoding='utf8')
            (root/'scripts/lib/modelarts.sh').write_text('''
configure_modelarts_distributed() { NNODES=3; NUM_NPUS=8; NODE_RANK=0; }
run_torchrun() {
  while [[ ! -f "${PROJECT}/guard-started" ]]; do sleep 0.05; done
  echo 'last sampler output'
  return "${TEST_TASK_CODE}"
}
''', encoding='utf8')
            (root/'python-stub.sh').write_text('''#!/bin/bash
set -eu
case "$1" in
  */npu_idle_guard.py)
    if [[ "${TEST_STUCK_GUARD}" == 1 ]]; then
      trap '' TERM
    else
      trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM
    fi
    touch "${PROJECT}/guard-started"
    while true; do sleep 0.05; done
    ;;
  */window_run_io.py)
    if [[ "$*" == *--watch* ]]; then
      trap 'exit 0' TERM
      while true; do sleep 0.05; done
    fi
    echo final-publish >> "${PROJECT}/order"
    exit "${TEST_PUBLISH_CODE}"
    ;;
  -)
    cat >/dev/null
    printf '{"exit_code":%s,"publication_failed":"%s"}\\n' "$3" "$4" > "$2/exit.json"
    ;;
  *) exit 99 ;;
esac
''', encoding='utf8')
            # chmod is relevant when these tests run on the Linux cluster/CI.
            (root/'python-stub.sh').chmod(0o755)
            env = dict(os.environ, SCENE_HELPER_STOP_TIMEOUT='1',
                       TEST_STUCK_GUARD=str(int(stuck_guard)),
                       TEST_TASK_CODE=str(task_code), TEST_PUBLISH_CODE=str(publish_code),
                       SCENE_NPU_IDLE_GUARD='1',
                       SCENE_SOURCE_NAMESPACE='source', SCENE_AUDIT_NAMESPACE='audit')
            result = subprocess.run([BASH, (root/'scripts/scale/52_audit_scene_flow_weights.sh').as_posix()],
                                    env=env, cwd=root, capture_output=True, text=True, timeout=15)
            logs = root/'output/audit/launcher/node0'
            record = json.loads((logs/'exit.json').read_text())
            return result, record, (logs/'pipeline.log').read_text(), (root/'order').read_text()

    def test_normal_cleanup_reaps_guard_before_publish_and_saves_tail(self):
        result, record, log, order = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(record['exit_code'], 0)
        self.assertLess(order.index('guard-stopped'), order.index('final-publish'))
        self.assertIn('last sampler output', log)
        self.assertIn('Scene replay launcher finished:', log)

    def test_blocked_guard_is_killed_instead_of_holding_node(self):
        result, record, log, order = self.run_launcher(stuck_guard=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(record['exit_code'], 0)
        self.assertIn('exceeded 1s; sending SIGKILL', log)
        self.assertIn('final-publish', order)

    def test_task_failure_is_preserved_through_cleanup(self):
        result, record, _log, _order = self.run_launcher(task_code=7, publish_code=9)
        self.assertEqual(result.returncode, 7, result.stdout+result.stderr)
        self.assertEqual(record['exit_code'], 7)
        self.assertEqual(record['publication_failed'], 'failed')

    def test_final_publication_failure_is_visible(self):
        result, record, _log, _order = self.run_launcher(publish_code=9)
        self.assertEqual(result.returncode, 73, result.stdout+result.stderr)
        self.assertEqual(record['exit_code'], 73)


if __name__ == '__main__':
    unittest.main()
