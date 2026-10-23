"""Coordinate independent per-node preparation commands without HCCL.

Each node runs its own command (usually one torchrun agent with eight teacher
workers). The TCP store propagates a failed worker group to the other nodes and
keeps successful nodes alive until all groups finish. No NPU memory is used.
"""
import argparse
from datetime import timedelta
import os
import signal
import subprocess
import time

import torch.distributed as dist


def stop_child(child, grace=10):
    if child is None or child.poll() is not None:
        return
    try:
        if os.name == 'posix':
            os.killpg(child.pid, signal.SIGTERM)
        else:
            child.terminate()
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'posix':
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            return
        child.wait(timeout=grace)


def run(command, store, rank, nodes, timeout, heartbeat=30, stop_grace=10):
    started = time.monotonic()
    child = None
    code = None
    reported = False
    next_heartbeat = started
    keys = [f'result{i}' for i in range(nodes)]
    try:
        child = subprocess.Popen(command, start_new_session=os.name == 'posix')
        while True:
            now = time.monotonic()
            if code is None:
                child_code = child.poll()
                if child_code is not None:
                    code = child_code if 0 <= child_code <= 255 else 1
                elif now-started > timeout:
                    print('[control preparation coordinator] command timed out', flush=True)
                    code = 124
                    stop_child(child, stop_grace)
            if code is not None and not reported:
                store.set(keys[rank], str(code))
                reported = True
            results = {}
            for i, key in enumerate(keys):
                if store.check([key]):
                    results[i] = int(store.get(key).decode())
            failed = next((value for value in results.values() if value), None)
            if failed is not None and code is None:
                print('[control preparation coordinator] peer failed; stopping local workers', flush=True)
                stop_child(child, stop_grace)
                code = failed
                store.set(keys[rank], str(code))
                reported = True
            if len(results) == nodes:
                code = failed or 0
                break
            if now >= next_heartbeat:
                print('[control preparation coordinator] node%d phase=%s finished=%d/%d elapsed=%.0fs' %
                      (rank, 'running' if code is None else 'waiting', len(results), nodes, now-started),
                      flush=True)
                next_heartbeat = now+heartbeat
            if now-started > timeout+2*stop_grace+120:
                raise TimeoutError('Missing preparation worker result after timeout')
            time.sleep(min(1., heartbeat))
        store.set(f'ack{rank}', '1')
        if rank == 0:
            store.wait([f'ack{i}' for i in range(nodes)], timedelta(seconds=120))
        return code
    except BaseException:
        # A spawn failure on a non-leader must not leave healthy peers waiting
        # for the full preparation timeout. A disconnected store is already a
        # peer-visible failure; preserve the original exception in that case.
        if not reported:
            try:
                store.set(keys[rank], '1')
            except Exception:
                pass
        raise
    finally:
        stop_child(child, stop_grace)
        print('[control preparation coordinator] node%d exit=%s' % (rank, code), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=172800)
    parser.add_argument('--heartbeat', type=float, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0 or args.heartbeat <= 0:
        parser.error('a command and positive timeout/heartbeat are required')
    def terminate(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    rank, nodes = int(os.environ['NODE_RANK']), int(os.environ['NNODES'])
    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                          nodes, rank == 0, timedelta(seconds=600), use_libuv=False)
    raise SystemExit(run(command, store, rank, nodes, args.timeout, args.heartbeat))


if __name__ == '__main__':
    main()
