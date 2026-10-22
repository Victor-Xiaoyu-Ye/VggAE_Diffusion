"""Exercise the real stage55 shell with local inference/IO/guard stubs.

These tests check distributed job orchestration and error propagation, not
VGGT compatibility, NPU kernels or dynamic-mask quality.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
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
class SceneTrackmaskLauncherTests(unittest.TestCase):
    def run_launcher(self, *, worker_code=0, staging_code=0, publish_code=0,
                     download_code=0, stuck_guard=False, missing_head=False,
                     node_rank=1, extra_env=None, real_staging=False,
                     local_checkpoint=False, use_default_namespace=False):
        temporary_root = REPO.parent if os.name == 'nt' else None
        with tempfile.TemporaryDirectory(prefix='trackmask launcher ', dir=temporary_root) as tmp:
            root = Path(tmp)
            for part in ('scripts/scale', 'scripts/lib', 'third_party/vggt_trackmask/vggt/models',
                         'third_party/vggt_trackmask/vggt/heads'):
                (root/part).mkdir(parents=True)
            for part in ('scripts/scale/55_audit_scene_trackmask.sh', 'scripts/lib/job_cleanup.sh'):
                shutil.copyfile(REPO/part, root/part)
            (root/'third_party/vggt_trackmask/vggt/models/vggt.py').touch()
            if not missing_head:
                (root/'third_party/vggt_trackmask/vggt/heads/track_head.py').touch()
            write_shell(root/'scripts/spatialvid_config.sh', '''
CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SCALE_ROOT="${PROJECT}/output"
SCALE_REMOTE_ROOT="${PROJECT}/primary"
SCALE_MIRROR_ROOT="${PROJECT}/mirror"
LOCAL_CACHE_ROOT="${PROJECT}/cache"
VGGAE_REF_ROOT="${PROJECT}/teacher assets"
STREAMVGGT_CKPT="${STREAMVGGT_CKPT:-${VGGAE_REF_ROOT}/StreamVGGT/checkpoints.pth}"
STREAMVGGT_URL="${STREAMVGGT_URL:-obs://fixture/primary.pth}"
STREAMVGGT_MIRROR_URL="${STREAMVGGT_MIRROR_URL:-obs://fixture/mirror.pth}"
PYTHON_BIN="${PROJECT}/python-stub.sh"
export PROJECT
''')
            write_shell(root/'scripts/lib/modelarts.sh', '''
configure_modelarts_distributed() { NNODES=3; NUM_NPUS=8; NODE_RANK=${TEST_NODE_RANK}; }
ensure_local_checkpoint() {
  local attempt
  for attempt in {1..200}; do
    [[ ! -f "${PROJECT}/guard-started" ]] || break
    sleep 0.02
  done
  [[ -f "${PROJECT}/guard-started" ]] || return 98
  printf 'checkpoint-staging\\n' >> "${PROJECT}/order"
  printf 'staging-arg=%s\\n' "$@" >> "${PROJECT}/arguments"
  if [[ "${TEST_STAGING_CODE}" != 0 ]]; then
    echo 'checkpoint staging failed'
    return "${TEST_STAGING_CODE}"
  fi
  mkdir -p "$(dirname "$1")"
  printf 'fixture full VGGT teacher\\n' > "$1"
}
''')
            if real_staging:
                write_shell(root/'scripts/lib/modelarts.sh',
                            (REPO/'scripts/lib/modelarts.sh').read_text()+'''
configure_modelarts_distributed() { NNODES=3; NUM_NPUS=8; NODE_RANK=${TEST_NODE_RANK}; }
''')
            if local_checkpoint:
                checkpoint = root/'teacher assets/StreamVGGT/checkpoints.pth'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b'existing full StreamVGGT fixture')
            write_shell(root/'python-stub.sh', '''#!/bin/bash
set -eu
case "$1" in
  */moxing_transfer.py)
    echo "transfer-source=$2" >> "${PROJECT}/arguments"
    if [[ "$2" == obs://fixture/primary.pth ]]; then
      printf 'partial invalid primary' > "$3"
      exit 22
    fi
    printf 'full StreamVGGT fixture from mirror' > "$3"
    ;;
  */npu_idle_guard.py)
    printf 'guard-arg=%s\\n' "$@" >> "${PROJECT}/guard-arguments"
    if [[ "${TEST_STUCK_GUARD}" == 1 ]]; then trap '' TERM
    else trap 'echo guard-stopped >> "${PROJECT}/order"; exit 0' TERM; fi
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
    exit "${TEST_PUBLISH_CODE}"
    ;;
  */audit_scene_trackmask.py)
    echo worker-started >> "${PROJECT}/order"
    printf 'worker-arg=%s\\n' "$@" >> "${PROJECT}/arguments"
    echo 'worker final output; DI_throughput=1.0'
    exit "${TEST_WORKER_CODE}"
    ;;
  */prepare_trackmask_teacher.py)
    for attempt in {1..200}; do
      [[ ! -f "${PROJECT}/guard-started" ]] || break
      sleep 0.02
    done
    [[ -f "${PROJECT}/guard-started" ]] || exit 98
    echo official-preparation >> "${PROJECT}/order"
    printf 'prepare-arg=%s\\n' "$@" >> "${PROJECT}/arguments"
    echo 'official preparation final output'
    exit "${TEST_DOWNLOAD_CODE}"
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
            env = dict(os.environ, SCENE_HELPER_STOP_TIMEOUT='1', SCENE_NPU_IDLE_GUARD='1',
                       SCENE_TRACKMASK_SOURCE_NAMESPACE='source', SCENE_TRACKMASK_NAMESPACE='audit',
                       TEST_NODE_RANK=str(node_rank), TEST_WORKER_CODE=str(worker_code),
                       TEST_DOWNLOAD_CODE=str(download_code),
                       TEST_STAGING_CODE=str(staging_code), TEST_PUBLISH_CODE=str(publish_code),
                       TEST_STUCK_GUARD=str(int(stuck_guard)))
            for key in tuple(env):
                if key.startswith(('VGGT_TRACKMASK_', 'STREAMVGGT_', 'SCENE_TRACKMASK_TEACHER_')):
                    env.pop(key)
            if use_default_namespace:
                env.pop('SCENE_TRACKMASK_NAMESPACE')
            env.update(extra_env or {})
            started = time.monotonic()
            result = subprocess.run([BASH, (root/'scripts/scale/55_audit_scene_trackmask.sh').as_posix()],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=20)
            elapsed = time.monotonic()-started
            default_namespace = ('scene_trackmask_stage54_v1' if
                                 env.get('SCENE_TRACKMASK_TEACHER_KIND') == 'vggt' else
                                 'scene_trackmask_stream_stage54_v2')
            namespace = env.get('SCENE_TRACKMASK_NAMESPACE', default_namespace)
            logs = root/f'output/{namespace}/launcher/node{node_rank}'
            self.assertTrue((logs/'exit.json').is_file(), result.stdout+result.stderr)
            def read(name):
                path = root/name
                return path.read_text() if path.exists() else ''
            return dict(result=result, elapsed=elapsed,
                        record=json.loads((logs/'exit.json').read_text()),
                        log=(logs/'pipeline.log').read_text(), order=read('order'),
                        arguments=read('arguments'), guard_arguments=read('guard-arguments'),
                        publish_arguments=read('publish-arguments'),
                        checkpoint=read('teacher assets/StreamVGGT/checkpoints.pth'))

    def test_independent_worker_shards_and_dual_paths_with_complete_log_tail(self):
        run = self.run_launcher()
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertEqual(run['record'], dict(exit_code=0, publication_failed=False))
        self.assertEqual(run['order'].count('worker-started'), 1)
        args = run['arguments']
        for pair in ('--node-rank\nworker-arg=1', '--nodes\nworker-arg=3',
                     '--seed\nworker-arg=101', '--step\nworker-arg=6000',
                     '--weights\nworker-arg=ema', '--device\nworker-arg=npu:0',
                     '--precision\nworker-arg=bfloat16', '--query-stride\nworker-arg=21',
                     '--allocated-devices\nworker-arg=8',
                     '--teacher-kind\nworker-arg=streamvggt'):
            self.assertIn('worker-arg='+pair, args)
        self.assertEqual(args.count('worker-arg=--source-root\n'), 2)
        self.assertEqual(args.count('worker-arg=--root\n'), 2)
        self.assertIn('/primary/source\n', args)
        self.assertIn('/mirror/source\n', args)
        self.assertNotIn('/source/eval/', args)
        self.assertIn('/teacher assets/StreamVGGT/checkpoints.pth', args)
        self.assertNotIn('worker-arg=--teacher-code\n', args)
        self.assertNotIn('official-preparation', run['order'])
        self.assertIn('guard-arg=--devices\nguard-arg=8', run['guard_arguments'])
        self.assertLess(run['order'].index('guard-started'), run['order'].index('checkpoint-staging'))
        self.assertLess(run['order'].index('checkpoint-staging'), run['order'].index('worker-started'))
        self.assertLess(run['order'].index('guard-stopped'), run['order'].index('final-publish'))
        self.assertLess(run['order'].index('publisher-stopped'), run['order'].index('final-publish'))
        self.assertIn('worker final output; DI_throughput=1.0', run['log'])
        self.assertIn('Scene trackmask launcher finished: node=1, exit_code=0', run['log'])
        self.assertIn('/primary/audit/launcher/node1', run['publish_arguments'])
        self.assertIn('/mirror/audit/launcher/node1', run['publish_arguments'])

    def test_worker_failure_survives_publication_failure(self):
        run = self.run_launcher(worker_code=17, publish_code=9)
        self.assertEqual(run['result'].returncode, 17)
        self.assertEqual(run['record'], dict(exit_code=17, publication_failed=True))
        self.assertNotIn('Scene trackmask worker completed:', run['log'])
        self.assertIn('worker final output', run['log'])

    def test_staging_failure_does_not_run_teacher_and_cleans_guard(self):
        run = self.run_launcher(staging_code=11)
        self.assertEqual(run['result'].returncode, 11)
        self.assertNotIn('worker-started', run['order'])
        self.assertIn('guard-stopped', run['order'])

    def test_explicit_vggt_missing_code_does_not_fall_back(self):
        run = self.run_launcher(missing_head=True, extra_env=dict(SCENE_TRACKMASK_TEACHER_KIND='vggt'))
        self.assertEqual(run['result'].returncode, 2)
        self.assertIn('Missing full-head VGGT code', run['log'])
        self.assertNotIn('checkpoint-staging', run['order'])
        self.assertNotIn('worker-started', run['order'])

    def test_stuck_guard_cleanup_is_bounded(self):
        run = self.run_launcher(stuck_guard=True)
        self.assertEqual(run['result'].returncode, 0)
        self.assertLess(run['elapsed'], 15)
        self.assertIn('NPU idle guard exceeded 1s; sending SIGKILL', run['log'])

    def test_publication_failure_is_not_success(self):
        run = self.run_launcher(publish_code=9)
        self.assertEqual(run['result'].returncode, 73)
        self.assertEqual(run['record'], dict(exit_code=73, publication_failed=True))
        self.assertEqual(run['order'].count('final-publish'), 2)

    def test_custom_cases_and_model_sources_are_passed_literally(self):
        run = self.run_launcher(extra_env=dict(
            SCENE_TRACKMASK_CASES='4,11,36', SCENE_TRACKMASK_SEED='202',
            SCENE_TRACKMASK_SOURCE_LOCAL='/fixture/manual downloads/source',
            SCENE_TRACKMASK_TEACHER_KIND='vggt',
            VGGT_TRACKMASK_CHECKPOINT_URL='obs://fixture/full-vggt/model.pt',
            VGGT_TRACKMASK_CHECKPOINT_MIRROR_URL='obs://fixture-mirror/full-vggt/model.pt'))
        self.assertEqual(run['result'].returncode, 0)
        self.assertIn('worker-arg=--cases\nworker-arg=4,11,36\n', run['arguments'])
        self.assertIn('worker-arg=--seed\nworker-arg=202\n', run['arguments'])
        self.assertIn('worker-arg=/fixture/manual downloads/source\n', run['arguments'])
        self.assertIn('staging-arg=obs://fixture/full-vggt/model.pt\n', run['arguments'])
        self.assertIn('staging-arg=obs://fixture-mirror/full-vggt/model.pt\n', run['arguments'])

    def test_explicit_official_preparation_runs_after_guard_before_worker(self):
        run = self.run_launcher(extra_env=dict(SCENE_TRACKMASK_TEACHER_KIND='vggt',
                                               VGGT_TRACKMASK_DOWNLOAD_OFFICIAL='1'))
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertNotIn('checkpoint-staging', run['order'])
        self.assertLess(run['order'].index('guard-started'), run['order'].index('official-preparation'))
        self.assertLess(run['order'].index('official-preparation'), run['order'].index('worker-started'))
        self.assertIn('prepare-arg=--destination\n', run['arguments'])
        self.assertIn('/teacher assets/VGGT-1B/model.safetensors\n', run['arguments'])
        self.assertIn('5,026,367,224 network bytes per node', run['log'])

    def test_official_preparation_failure_cleans_guard_without_worker(self):
        run = self.run_launcher(download_code=19, extra_env=dict(SCENE_TRACKMASK_TEACHER_KIND='vggt',
                                                                VGGT_TRACKMASK_DOWNLOAD_OFFICIAL='1'))
        self.assertEqual(run['result'].returncode, 19)
        self.assertNotIn('worker-started', run['order'])
        self.assertIn('guard-stopped', run['order'])
        self.assertIn('official preparation final output', run['log'])

    def test_official_preparation_rejects_wrong_extension_before_transfer(self):
        run = self.run_launcher(extra_env=dict(SCENE_TRACKMASK_TEACHER_KIND='vggt',
                                               VGGT_TRACKMASK_DOWNLOAD_OFFICIAL='1',
                                               VGGT_TRACKMASK_CHECKPOINT='/fixture/model.pt'))
        self.assertEqual(run['result'].returncode, 2)
        self.assertNotIn('official-preparation', run['order'])
        self.assertNotIn('worker-started', run['order'])
        self.assertIn('.safetensors destination', run['log'])

    def test_stream_ignores_public_download_opt_in_and_uses_new_namespace(self):
        run = self.run_launcher(use_default_namespace=True, extra_env=dict(
            VGGT_TRACKMASK_DOWNLOAD_OFFICIAL='1'))
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertNotIn('official-preparation', run['order'])
        self.assertIn('checkpoint-staging', run['order'])
        self.assertIn('/output/scene_trackmask_stream_stage54_v2\n', run['arguments'])
        self.assertIn('Ignoring VGGT_TRACKMASK_DOWNLOAD_OFFICIAL', run['log'])

    def test_stream_default_urls_and_code_override_are_passed_literally(self):
        run = self.run_launcher(extra_env=dict(
            STREAMVGGT_CKPT='teacher overrides/existing checkpoint.pth',
            STREAMVGGT_URL='obs://fixture/custom primary.pth',
            STREAMVGGT_MIRROR_URL='obs://fixture/custom mirror.pth',
            SCENE_TRACKMASK_TEACHER_CODE='/fixture/teacher source'))
        self.assertEqual(run['result'].returncode, 0)
        self.assertIn('staging-arg=obs://fixture/custom primary.pth\n', run['arguments'])
        self.assertIn('staging-arg=obs://fixture/custom mirror.pth\n', run['arguments'])
        self.assertIn('worker-arg=--teacher-checkpoint\nworker-arg=teacher overrides/existing checkpoint.pth\n',
                      run['arguments'])
        self.assertIn('worker-arg=--teacher-code\nworker-arg=/fixture/teacher source\n', run['arguments'])

    def test_real_helper_uses_mirror_after_failed_primary(self):
        run = self.run_launcher(real_staging=True)
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertIn('transfer-source=obs://fixture/primary.pth\n', run['arguments'])
        self.assertIn('transfer-source=obs://fixture/mirror.pth\n', run['arguments'])
        self.assertLess(run['arguments'].index('transfer-source=obs://fixture/primary.pth'),
                        run['arguments'].index('transfer-source=obs://fixture/mirror.pth'))
        self.assertEqual(run['checkpoint'], 'full StreamVGGT fixture from mirror')
        self.assertNotIn('official-preparation', run['order'])

    def test_real_helper_preserves_existing_local_checkpoint_without_transfer(self):
        run = self.run_launcher(real_staging=True, local_checkpoint=True)
        self.assertEqual(run['result'].returncode, 0, run['result'].stdout+run['result'].stderr)
        self.assertNotIn('transfer-source=', run['arguments'])
        self.assertEqual(run['checkpoint'], 'existing full StreamVGGT fixture')
        self.assertNotIn('official-preparation', run['order'])

    def test_stream_rejects_web_source_without_running_downloader(self):
        run = self.run_launcher(extra_env=dict(STREAMVGGT_URL='https://example.com/model.pth'))
        self.assertEqual(run['result'].returncode, 2)
        self.assertNotIn('checkpoint-staging', run['order'])
        self.assertNotIn('official-preparation', run['order'])
        self.assertNotIn('worker-started', run['order'])
        self.assertIn('StreamVGGT sources must be existing local/OBS/S3 assets', run['log'])


if __name__ == '__main__':
    unittest.main()
