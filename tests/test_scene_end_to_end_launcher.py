"""Execute actual stage57 Bash orchestration with local cluster-operation stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
GIT_BASH=Path(r'C:\Program Files\Git\bin\bash.exe')
BASH=str(GIT_BASH) if os.name=='nt' and GIT_BASH.is_file() else shutil.which('bash')


def shell(path,text):
    with path.open('w',encoding='utf8',newline='\n') as stream: stream.write(text)
    path.chmod(0o755)


@unittest.skipUnless(BASH,'Bash required')
class EndToEndLauncherTests(unittest.TestCase):
    def launch(self,stage='all',staging=0,work=0,merge=0,publication=0,node=0):
        with tempfile.TemporaryDirectory(prefix='audit launcher ',dir=REPO.parent) as tmp:
            root=Path(tmp)
            for directory in ('scripts/scale','scripts/lib','configs'): (root/directory).mkdir(parents=True)
            for path in ('scripts/scale/57_audit_scene_end_to_end.sh','scripts/lib/job_cleanup.sh'):
                shutil.copyfile(REPO/path,root/path)
            (root/'configs/scene_end_to_end_audit_v1.json').write_text(json.dumps({'schema':'fixture','cases':[0,4]}))
            (root/'scripts/window_run_io.py').write_text('''
import json
from pathlib import Path
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
''',encoding='utf8')
            shell(root/'scripts/spatialvid_config.sh','''
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
export PROJECT
''')
            shell(root/'scripts/lib/modelarts.sh','''
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
            shell(root/'python-stub.sh','''#!/bin/bash
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
  echo "work-port=${MASTER_PORT}" >> "${PROJECT}/arguments"
  shift; while [[ "$1" != -- ]]; do shift; done; shift
  "$@" ;;
 */coordinated_diagnostic.py)
  echo "merge-port=${MASTER_PORT}" >> "${PROJECT}/arguments"
  shift
  if [[ "${NODE_RANK}" == 0 ]]; then "$@"; else exit "${TEST_MERGE}"; fi ;;
 */audit_scene_end_to_end.py)
  printf 'auditarg=%s\n' "$@" >> "${PROJECT}/arguments"
  phase=''
  while (( $# )); do
   if [[ "$1" == --stage ]]; then phase=$2; shift; fi
   shift
  done
  echo "${phase}" >> "${PROJECT}/order"
  if [[ "${phase}" == work ]]; then exit "${TEST_WORK}"; else exit "${TEST_MERGE}"; fi ;;
 -) "$REAL_PYTHON" "$@" ;;
 *) echo "Unexpected call: $*" >&2; exit 99 ;;
esac
''')
            env=dict(os.environ,REAL_PYTHON=Path(sys.executable).as_posix(),TEST_NODE=str(node),
                TEST_STAGING=str(staging),TEST_WORK=str(work),TEST_MERGE=str(merge),
                TEST_PUBLICATION=str(publication),SCENE_AUDIT_STAGE=stage,SCENE_AUDIT_NAMESPACE='fixture',
                SCENE_AUDIT_SOURCE_NAMESPACE='source',SCENE_NPU_IDLE_GUARD='1',SCENE_HELPER_STOP_TIMEOUT='1',MASTER_PORT='29957')
            for key in ('SCENE_AUDIT_CONFIG','SCENE_AUDIT_SOURCE_LOCAL','SCENE_AUDIT_SOURCE_PRIMARY',
                        'SCENE_AUDIT_SOURCE_MIRROR','PYTHONPATH'): env.pop(key,None)
            result=subprocess.run([BASH,(root/'scripts/scale/57_audit_scene_end_to_end.sh').as_posix()],
                cwd=root,env=env,capture_output=True,text=True,timeout=30)
            logs=root/'output/fixture/launcher'/f'node{node}'
            read=lambda name: (root/name).read_text() if (root/name).is_file() else ''
            records={p.stem:json.loads(p.read_text()) for p in logs.glob('*.json')} if logs.exists() else {}
            return dict(result=result,order=read('order').splitlines(),arguments=read('arguments'),records=records)

    def test_full_work_then_merge_with_guard_and_distinct_ports(self):
        r=self.launch()
        self.assertEqual(r['result'].returncode,0,r['result'].stdout+r['result'].stderr)
        self.assertLess(r['order'].index('guard-started'),r['order'].index('staging'))
        self.assertLess(r['order'].index('work'),r['order'].index('merge'))
        self.assertLess(r['order'].index('guard-stopped'),r['order'].index('published'))
        self.assertIn('work-port=29958',r['arguments']); self.assertIn('merge-port=29959',r['arguments'])
        self.assertIn('guardarg=--devices\nguardarg=8',r['arguments'])
        self.assertIn('auditarg=--device\nauditarg=npu:0',r['arguments'])
        self.assertEqual(r['arguments'].count('auditarg=--source-root'),4)
        self.assertEqual(r['arguments'].count('auditarg=--root'),4)
        self.assertEqual(r['records']['exit']['exit_code'],0)
        self.assertEqual(r['records']['effective_config']['cases'],[0,4])

    def test_worker_failure_still_merges_and_keeps_original_exit(self):
        r=self.launch(work=7,merge=9)
        self.assertEqual(r['result'].returncode,7)
        self.assertIn('merge',r['order'])
        self.assertEqual(r['records']['phase_work']['exit_code'],7)
        self.assertEqual(r['records']['phase_merge']['exit_code'],9)
        self.assertEqual(r['records']['exit']['exit_code'],7)

    def test_teacher_staging_failure_does_not_skip_independent_diagnostics(self):
        r=self.launch(staging=6)
        self.assertEqual(r['result'].returncode,6)
        self.assertIn('work',r['order']);self.assertIn('merge',r['order'])
        self.assertEqual(r['records']['phase_checkpoint']['exit_code'],6)

    def test_merge_failure_not_reported_as_success(self):
        r=self.launch(merge=8)
        self.assertEqual(r['result'].returncode,8)
        self.assertEqual(r['records']['exit']['exit_code'],8)

    def test_publication_failure_preserved(self):
        r=self.launch(publication=3)
        self.assertEqual(r['result'].returncode,73)
        self.assertTrue(r['records']['exit']['publication_failed'])

    def test_nonleader_runs_work_and_waits_for_merge(self):
        r=self.launch(node=2)
        self.assertEqual(r['result'].returncode,0,r['result'].stdout+r['result'].stderr)
        self.assertIn('work',r['order']); self.assertNotIn('merge',r['order'])
        self.assertEqual(r['records']['phase_work']['node'],2)
        self.assertEqual(r['records']['phase_merge']['exit_code'],0)

    def test_dry_has_no_staging_or_output(self):
        r=self.launch(stage='dry')
        self.assertEqual(r['result'].returncode,0)
        self.assertEqual(r['order'],[]);self.assertEqual(r['records'],{})
        self.assertIn('no HCCL or training',r['result'].stdout)


if __name__=='__main__': unittest.main()
