"""CPU tests for telemetry decisions and helper lifetime, not NPU utilization."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.npu_idle_guard import (IdlePolicy, Reporter, bind_parent,
                                   device_mapping, usage_percent, worker, stop_workers)


MAPPING = '''NPU ID   Chip ID   Chip Logic ID   Chip Name
2     0      0       Ascend 910B3
2     1      -       Mcu
4     0      1       Ascend 910B3
6     0      2       Ascend 910B3
'''


def cooperative_child(stop, ready):
    ready.set()
    stop.wait(60)


def blocked_child(ready):
    ready.set()
    import time
    time.sleep(60)


class StopAfterWaits:
    def __init__(self, count):
        self.count = count
    def is_set(self):
        return self.count <= 0
    def wait(self, _seconds):
        self.count -= 1
        return self.is_set()


class NPUIdleGuardTests(unittest.TestCase):
    def test_runtime_visible_devices_map_to_physical_chips(self):
        self.assertEqual(device_mapping(MAPPING, 2), [(2, 0), (4, 0)])
        self.assertEqual(device_mapping(MAPPING, 2, '1,2'), [(4, 0), (6, 0)])
        for visible in ('1, 2', '1,1', '9,10', ''):
            with self.assertRaises(ValueError):
                device_mapping(MAPPING, 2, visible)
        with self.assertRaises(ValueError):
            device_mapping(MAPPING, 4)

    def test_parse_aicore_only_and_reject_ambiguous_or_missing_values(self):
        self.assertEqual(usage_percent('HBM Usage Rate(%) : 90\nAicore Usage Rate(%) : 1.5'), 1.5)
        for text in ('Aicore Usage Rate(%) : N/A', 'Memory Usage Rate(%) : 0',
                     'Aicore Usage Rate(%) : 101', 'Aicore Usage Rate(%) : 0\nAicore Usage Rate(%) : 30'):
            with self.assertRaises(ValueError):
                usage_percent(text)

    def test_sustained_low_utilization_period_and_resumed_training(self):
        policy = IdlePolicy(2, 600, 60)
        self.assertFalse(policy.observe(0, 0))
        self.assertFalse(policy.observe(599, 1))
        self.assertTrue(policy.observe(600, 0))
        policy.last_pulse = 600
        self.assertFalse(policy.observe(630, 0))
        self.assertTrue(policy.observe(660, 0))
        self.assertFalse(policy.observe(661, 2))
        self.assertFalse(policy.observe(700, 0))
        self.assertFalse(policy.observe(1299, 1))
        self.assertTrue(policy.observe(1300, 1))

    def test_unknown_telemetry_never_becomes_idle_evidence(self):
        for invalid in (None, float('nan'), float('inf'), -1, 101):
            policy = IdlePolicy(2, 600, 60)
            policy.observe(0, 0)
            self.assertFalse(policy.observe(600, invalid))
            self.assertFalse(policy.observe(601, 0))

    def test_busy_or_unreadable_devices_do_not_allocate_workload(self):
        options = SimpleNamespace(output='unused', threshold=2, idle_seconds=600, period=60, poll=30)
        for result in ('Aicore Usage Rate(%) : 80', subprocess.TimeoutExpired('npu-smi', 10)):
            with patch('scripts.npu_idle_guard.bind_parent'), patch('scripts.npu_idle_guard.Reporter'), \
                 patch('scripts.npu_idle_guard.smi', side_effect=[result] if isinstance(result, Exception) else None,
                       return_value=result), patch('scripts.npu_idle_guard.MatmulPulse') as workload:
                worker(0, 2, 0, options, StopAfterWaits(1), os.getppid())
                workload.assert_not_called()

    def test_idle_workload_is_released_on_resumed_training(self):
        options = SimpleNamespace(output='unused', threshold=2, idle_seconds=0, period=60,
                                  poll=30, burst_seconds=10, matrix_size=2048)
        with patch('scripts.npu_idle_guard.bind_parent'), patch('scripts.npu_idle_guard.Reporter'), \
             patch('scripts.npu_idle_guard.smi', side_effect=['Aicore Usage Rate(%) : 0',
                                                           'Aicore Usage Rate(%) : 80']), \
             patch('scripts.npu_idle_guard.MatmulPulse') as workload:
            workload.return_value.run.return_value = {'seconds': 10, 'matmuls': 32}
            worker(0, 2, 0, options, StopAfterWaits(2), os.getppid())
            workload.assert_called_once_with(0, 2048)
            workload.return_value.run.assert_called_once()
            workload.return_value.close.assert_called_once()

    def test_parent_mismatch_refuses_orphan_work(self):
        with patch('scripts.npu_idle_guard.sys.platform', 'win32'), \
             patch('scripts.npu_idle_guard.os.getppid', return_value=123):
            with self.assertRaisesRegex(RuntimeError, 'Owner process'):
                bind_parent(456)

    def test_status_and_history_are_separate_from_training_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = Reporter(tmp, 'device0')
            reporter.write('pulse', utilization_before=0)
            reporter.write('stopped')
            text = (Path(tmp)/'device0.jsonl').read_text()
            self.assertIn('pulse', text)
            self.assertIn('stopped', (Path(tmp)/'device0.json').read_text())
            self.assertNotIn('DI_throughput', text)

    def test_shutdown_reaps_cooperative_and_blocked_helpers(self):
        context = mp.get_context('spawn')
        stop = context.Event()
        ready = [context.Event(), context.Event()]
        workers = [context.Process(target=cooperative_child, args=(stop, ready[0])),
                   context.Process(target=blocked_child, args=(ready[1],))]
        try:
            for process in workers:
                process.start()
            self.assertTrue(all(event.wait(15) for event in ready))
        finally:
            stop_workers(workers, stop, grace=.1)
        self.assertTrue(all(not process.is_alive() for process in workers))


if __name__ == '__main__':
    unittest.main()
