from datetime import timedelta
import sys
from concurrent.futures import ThreadPoolExecutor
import unittest
import torch.distributed as dist
from scripts.coordinated_diagnostic import run


class CoordinatorTests(unittest.TestCase):
    def test_followers_wait_and_receive_actual_exit_code(self):
        for code in (0, 7):
            with self.subTest(code=code):
                leader = dist.TCPStore('127.0.0.1', 0, 2, True,
                                       timedelta(seconds=5), wait_for_workers=False, use_libuv=False)
                follower = dist.TCPStore('127.0.0.1', leader.port, 2, False,
                                         timedelta(seconds=5), use_libuv=False)
                with ThreadPoolExecutor(2) as pool:
                    worker = pool.submit(run, [], follower, 1, 2, 5, .01)
                    command = [sys.executable, '-c',
                               'import time;time.sleep(.1);raise SystemExit(%d)' % code]
                    self.assertEqual(run(command, leader, 0, 2, 5, .01), code)
                    self.assertEqual(worker.result(timeout=5), code)

    def test_missing_executable_publishes_failure(self):
        leader = dist.TCPStore('127.0.0.1', 0, 1, True, timedelta(seconds=5), use_libuv=False)
        with self.assertRaises(FileNotFoundError):
            run(['nonexistent-vggae-executable'], leader, 0, 1, 5, .01)
        self.assertEqual(leader.get('result'), b'1')
