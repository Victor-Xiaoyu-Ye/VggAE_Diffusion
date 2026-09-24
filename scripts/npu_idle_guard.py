"""Job-scoped Ascend idle guard; separate from model state and training metrics.

Monitor npu-smi without allocating HBM. After sustained low utilization, lazily
create a small independent FP16 matmul workload. Unknown telemetry is never
treated as zero. Only devices used by this job are selected. No HCCL groups.
"""
import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def smi(*args):
    result = subprocess.run(['npu-smi', 'info', *args], check=True,
                            capture_output=True, text=True, timeout=10)
    return result.stdout


def device_mapping(text, count, visible=None):
    mapping = {}
    for line in text.splitlines():
        match = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+Ascend\b', line)
        if match:
            card, chip, logical = map(int, match.groups())
            if logical in mapping:
                raise ValueError('Duplicate logical device in npu-smi mapping')
            mapping[logical] = (card, chip)
    if visible is not None:
        if not re.fullmatch(r'\d+(,\d+)*', visible):
            raise ValueError('Invalid ASCEND_RT_VISIBLE_DEVICES; cannot safely map devices')
        ids = list(map(int, visible.split(',')))
    else:
        ids = list(range(count))
    if len(ids) < count or len(set(ids)) != len(ids):
        raise ValueError('Insufficient or duplicate visible devices')
    if any(i not in mapping for i in ids[:count]):
        raise ValueError('Job device missing from npu-smi mapping')
    # Return local torch index -> physical card/chip, including runtime remap.
    return [mapping[i] for i in ids[:count]]


def usage_percent(text):
    values = re.findall(r'Aicore\s+Usage\s+Rate\s*\(%\)\s*:\s*(\d+(?:\.\d+)?)', text, re.I)
    if len(values) != 1:
        raise ValueError('Missing or ambiguous per-chip Aicore utilization')
    value = float(values[0])
    if not 0 <= value <= 100:
        raise ValueError('Invalid utilization range')
    return value


class IdlePolicy:
    def __init__(self, threshold, idle_seconds, period):
        self.threshold, self.idle_seconds, self.period = threshold, idle_seconds, period
        self.low_since, self.last_pulse = None, -float('inf')

    def observe(self, now, usage):
        if usage is None or not math.isfinite(usage) or not 0 <= usage < self.threshold:
            self.low_since = None
            return False
        if self.low_since is None:
            self.low_since = now
        return now-self.low_since >= self.idle_seconds and now-self.last_pulse >= self.period


def bind_parent(parent_pid):
    """Parent death also terminates blocked driver calls on Linux (no orphans)."""
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'Cannot bind idle guard to job lifetime')
    if os.getppid() != parent_pid:
        raise RuntimeError('Owner process has already exited')


class Reporter:
    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.root.mkdir(parents=True, exist_ok=True)
        self.previous, self.last_history = None, -float('inf')

    def write(self, state, **values):
        row = dict(state=state, unix_time=time.time(), pid=os.getpid(), **values)
        partial = self.root/(self.name+'.tmp')
        partial.write_text(json.dumps(row)+'\n', encoding='utf8')
        os.replace(partial, self.root/(self.name+'.json'))
        changed = state != self.previous
        if changed or time.monotonic()-self.last_history >= 60:
            with (self.root/(self.name+'.jsonl')).open('a', encoding='utf8') as handle:
                handle.write(json.dumps(row)+'\n')
            self.last_history = time.monotonic()
        if changed or state in ('pulse', 'warning', 'error'):
            print('[NPU idle guard] '+json.dumps(dict(device=self.name, **row)), flush=True)
        self.previous = state


class MatmulPulse:
    def __init__(self, index, size):
        import torch
        import torch_npu  # noqa: F401
        torch.set_num_threads(1)
        if index >= torch.npu.device_count():
            raise ValueError('Idle guard index outside runtime-visible devices')
        self.torch, self.device = torch, torch.device(f'npu:{index}')
        torch.npu.set_device(self.device)
        # 3 * size^2 * 2 bytes: 24 MiB at 2048; runtime/workspace are additional.
        self.a = torch.full((size, size), .01, dtype=torch.float16, device=self.device)
        self.b = torch.full_like(self.a, .01)
        self.out = torch.empty_like(self.a)

    def run(self, seconds, stop):
        start, groups = time.monotonic(), 0
        with self.torch.no_grad():
            while not stop.is_set() and time.monotonic()-start < seconds:
                # Bound queued work so SIGTERM does not wait for a giant queue.
                for _ in range(16):
                    self.torch.mm(self.a, self.b, out=self.out)
                self.torch.npu.synchronize(self.device)
                groups += 1
        return dict(seconds=time.monotonic()-start, matmuls=groups*16,
                    allocated_mib=self.torch.npu.memory_allocated(self.device)/1024**2,
                    reserved_mib=self.torch.npu.memory_reserved(self.device)/1024**2)

    def close(self):
        self.a = self.b = self.out = None
        self.torch.npu.empty_cache()


def worker(index, card, chip, options, stop, parent_pid):
    bind_parent(parent_pid)
    # SIGTERM from the coordinator terminates only this helper, never training.
    reporter = Reporter(options.output, f'device{index}')
    policy = IdlePolicy(options.threshold, options.idle_seconds, options.period)
    workload = None
    try:
        reporter.write('monitoring', card=card, chip=chip, hbm_initialized=False)
        while not stop.is_set():
            if os.getppid() != parent_pid:
                break
            try:
                usage = usage_percent(smi('-t', 'usages', '-i', str(card), '-c', str(chip)))
                due = policy.observe(time.monotonic(), usage)
                if usage >= options.threshold and workload is not None:
                    workload.close()
                    workload = None
                reporter.write('low' if usage < options.threshold else 'busy', utilization=usage,
                               card=card, chip=chip,
                               low_seconds=0 if policy.low_since is None else time.monotonic()-policy.low_since)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                policy.observe(time.monotonic(), None)
                reporter.write('warning', error=repr(exc), protection='unverified; no blind workload')
                stop.wait(max(60, options.poll))
                continue
            if due and not stop.is_set():
                pulse_start = time.monotonic()
                reporter.write('pulse', utilization_before=usage)
                if workload is None:
                    workload = MatmulPulse(index, options.matrix_size)
                result = workload.run(options.burst_seconds, stop)
                policy.last_pulse = pulse_start
                reporter.write('cooldown', **result)
                # Sample only AFTER our burst and a quiet interval: the guard
                # must not mistake its own utilization for resumed training.
                stop.wait(max(options.poll, options.period-(time.monotonic()-pulse_start)))
            else:
                stop.wait(options.poll)
        reporter.write('stopped')
    except Exception as exc:
        reporter.write('error', error=repr(exc), protection='inactive')
        raise


def stop_workers(workers, stop, grace=5):
    stop.set()
    deadline = time.monotonic()+grace
    for process in workers:
        process.join(max(0, deadline-time.monotonic()))
    for process in workers:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic()+2
    for process in workers:
        process.join(max(0, deadline-time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=2.)
    parser.add_argument('--idle-seconds', type=float, default=600.)
    parser.add_argument('--poll', type=float, default=30.)
    parser.add_argument('--period', type=float, default=60.)
    parser.add_argument('--burst-seconds', type=float, default=10.)
    parser.add_argument('--matrix-size', type=int, default=2048)
    options = parser.parse_args()
    if (options.devices < 1 or not 0 < options.threshold <= 100 or
        not 0 < options.burst_seconds < options.period or options.poll <= 0 or
        options.idle_seconds < 0 or not 128 <= options.matrix_size <= 4096 or
        not all(math.isfinite(x) for x in (options.threshold, options.idle_seconds,
                                          options.poll, options.period, options.burst_seconds))):
        parser.error('Invalid idle guard bounds')
    bind_parent(options.parent_pid)
    context = mp.get_context('spawn')
    stop = context.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    reporter = Reporter(options.output, 'supervisor')
    workers, disabled = [], set()
    failed_setup = False
    try:
        mapping = device_mapping(smi('-m'), options.devices, os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
        reporter.write('started', mapping=mapping, settings=vars(options))
        for index, (card, chip) in enumerate(mapping):
            process = context.Process(target=worker, args=(index, card, chip, options, stop, os.getpid()))
            process.start()
            workers.append(process)
        last_report = 0.
        while not stop.wait(1):
            if os.getppid() != options.parent_pid:
                break
            failed = {i for i, p in enumerate(workers) if p.exitcode is not None}
            if failed and (failed != disabled or time.monotonic()-last_report >= 300):
                reporter.write('warning', inactive_devices=sorted(failed),
                               message='Training continues, but idle protection is incomplete')
                last_report = time.monotonic()
            disabled = failed
    except Exception as exc:
        failed_setup = True
        reporter.write('error', error=repr(exc), protection='inactive; training is not blocked')
        return 1
    finally:
        stop_workers(workers, stop)
        reporter.write('stopped', inactive_devices=sorted(disabled), failed_setup=failed_setup)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
