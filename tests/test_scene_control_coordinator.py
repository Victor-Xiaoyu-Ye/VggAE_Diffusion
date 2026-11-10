"""Exercise real subprocess completion, peer failure and timeout cleanup."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch

from scripts.coordinated_scene_controls import run


class SharedStore:
    """Thread-safe TCPStore API fixture; no NPU or network required."""
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.values[key] = str(value).encode()

    def check(self, keys):
        with self.lock:
            return all(key in self.values for key in keys)

    def get(self, key):
        with self.lock:
            return self.values[key]

    def wait(self, keys, timeout):
        deadline = time.monotonic()+timeout.total_seconds()
        while not self.check(keys):
            if time.monotonic() > deadline:
                raise TimeoutError(keys)
            time.sleep(.01)


class SceneControlCoordinatorTests(unittest.TestCase):
    def pair(self, scripts, timeout=4):
        store = SharedStore()
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(run, [sys.executable, '-c', script], store,
                                 rank, 2, timeout, .02, .2)
                     for rank, script in enumerate(scripts)]
            result = [task.result(timeout=8) for task in tasks]
        self.assertLess(time.monotonic()-started, 6)
        self.assertTrue(store.check(['ack0', 'ack1']))
        return result

    def test_early_success_waits_for_peer(self):
        self.assertEqual(self.pair(['pass', 'import time; time.sleep(.2)']), [0, 0])

    def test_failed_worker_stops_sleeping_peer(self):
        self.assertEqual(self.pair(['raise SystemExit(7)', 'import time; time.sleep(30)']), [7, 7])

    def test_timeout_is_visible_and_children_are_stopped(self):
        self.assertEqual(self.pair(['import time; time.sleep(30)']*2, .2), [124, 124])

    def test_spawn_error_completes_failure_protocol(self):
        store = SharedStore()
        self.assertEqual(run(['definitely-not-a-valid-scene-preparation-executable'],
                             store,0,1,1,.02,.2),1)
        self.assertEqual(store.get('result0'), b'1')
        self.assertTrue(store.check(['ack0']))

    def test_disconnect_after_local_success_never_logs_phase_success(self):
        store=SharedStore()
        child=Mock(); child.poll.return_value=0
        text=io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            status=Path(temp)/'status.json'
            with patch('scripts.coordinated_scene_controls.subprocess.Popen',return_value=child), \
                    patch.object(store,'check',side_effect=ConnectionError('store owner exited')),redirect_stdout(text):
                with self.assertRaises(ConnectionError):
                    run(['fixture'],store,1,3,1,.02,.2,phase='cache',status_file=status)
            value=json.loads(status.read_text())
        self.assertEqual(value['local_exit'],0)
        self.assertEqual(value['phase_exit'],1)
        self.assertFalse(value['all_results_observed'])
        self.assertEqual(value['state'],'failed')
        self.assertIn('exit=1 local_exit=0',text.getvalue())

    def test_status_write_error_does_not_abort_result_ack(self):
        store=SharedStore()
        with patch.object(Path,'write_text',side_effect=OSError('disk full')):
            self.assertEqual(run([sys.executable,'-c','pass'],store,0,1,2,.02,.2,
                                 status_file='fixture.json'),0)
        self.assertTrue(store.check(['ack0']))

    def test_signal_exit_preserves_raw_cause_and_child_phase(self):
        # A POSIX subprocess killed by SIGSEGV reports -11, not exit status 1.
        # Simulate that Popen result on all test hosts without crashing Python.
        store = SharedStore()
        child = Mock(pid=4321); child.poll.return_value = -11
        text = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp)/'status.json'
            with patch('scripts.coordinated_scene_controls.subprocess.Popen',
                       return_value=child), redirect_stdout(text):
                code = run(['python', 'refine_vace_edit_state.py', '--phase', 'summarize'],
                           store, 0, 1, 1, .02, .2, phase='stage66_30', status_file=status)
            value = json.loads(status.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(value['child_returncode'], -11)
        self.assertEqual(value['child_signal'], 'SIGSEGV')
        self.assertEqual(value['child_pid'], 4321)
        self.assertEqual(value['job']['phase'], 'summarize')
        self.assertEqual(value['termination_reason'], 'signal')
        self.assertTrue(value['all_results_observed'])
        self.assertIn('child_returncode=-11 child_signal=SIGSEGV', text.getvalue())

    def test_regular_exit_does_not_masquerade_as_signal(self):
        store = SharedStore()
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp)/'status.json'
            self.assertEqual(run([sys.executable, '-c', 'raise SystemExit(7)'],
                                 store, 0, 1, 3, .02, .2, status_file=status), 7)
            value = json.loads(status.read_text())
        self.assertEqual(value['child_returncode'], 7)
        self.assertIsNone(value['child_signal'])
        self.assertEqual(value['termination_reason'], 'exited')
        self.assertEqual(store.get('result0'), b'7')


if __name__ == '__main__':
    unittest.main()
