"""Coordinate independent per-node preparation commands without HCCL.

Each node runs its own command (usually one torchrun agent with eight teacher
workers). The TCP store propagates a failed worker group to the other nodes and
keeps successful nodes alive until all groups finish. No NPU memory is used.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

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


def run(command, store, rank, nodes, timeout, heartbeat=30, stop_grace=10,
        phase='preparation', status_file=None):
    started = time.monotonic()
    child = None
    local_code = None
    phase_code = None
    reported = False
    confirmed = False
    local_error = None
    coordinator_error = None
    next_heartbeat = started
    keys = [f'result{i}' for i in range(nodes)]
    results = {}
    job=dict(program=Path(str(command[0])).name if command else None,
             entry=next((Path(str(arg)).name for arg in command if str(arg).endswith(('.py','.sh'))),None))
    for flag in ('--stage','--arm','--stop-after'):
        if flag in command and command.index(flag)+1<len(command):
            job[flag[2:]]=str(command[command.index(flag)+1])

    def status(state):
        value=dict(phase=phase,job=job,node=rank,state=state,local_exit=local_code,
            phase_exit=phase_code,all_results_observed=confirmed,results=results,
            local_error=local_error,coordinator_error=coordinator_error,
            hostname=socket.gethostname(),pid=os.getpid(),
            master_addr=os.environ.get('MASTER_ADDR'),master_port=os.environ.get('MASTER_PORT'),
            elapsed_seconds=time.monotonic()-started,time=time.time())
        if status_file:
            try:
                path=Path(status_file); path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
                temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
                os.replace(temp,path)
            except OSError as exc:
                # Logging must not tear down the node0 server before the peers
                # receive the actual command result or hide the original error.
                print(f'[control preparation coordinator] status write warning: {exc!r}',flush=True)

    try:
        status('starting')
        try:
            child = subprocess.Popen(command, start_new_session=os.name == 'posix')
        except Exception:
            # In particular, the node0 TCPStore owner must NOT unwind here.
            # Publish the launch failure through the normal result/ack protocol
            # so every live peer can read it before the server is destroyed.
            local_code = 1
            local_error = traceback.format_exc()
            print(f'[control preparation coordinator] node{rank} phase={phase} child launch failed:\n{local_error}',flush=True)
        while True:
            now = time.monotonic()
            if local_code is None:
                child_code = child.poll()
                if child_code is not None:
                    local_code = child_code if 0 <= child_code <= 255 else 1
                elif now-started > timeout:
                    print('[control preparation coordinator] command timed out', flush=True)
                    local_code = 124
                    stop_child(child, stop_grace)
            if local_code is not None and not reported:
                store.set(keys[rank], str(local_code))
                reported = True
                status('local_finished_waiting_for_peers')
            for i, key in enumerate(keys):
                # Result keys are immutable. Do not keep re-reading results
                # already received while waiting for slower nodes.
                if i not in results and store.check([key]):
                    results[i] = int(store.get(key).decode())
            failed = next((results[i] for i in sorted(results) if results[i]), None)
            if failed is not None and local_code is None:
                print('[control preparation coordinator] peer failed; stopping local workers', flush=True)
                stop_child(child, stop_grace)
                local_code = failed
                store.set(keys[rank], str(local_code))
                reported = True
            if len(results) == nodes:
                phase_code = failed or 0
                confirmed = True
                break
            if now >= next_heartbeat:
                print('[control preparation coordinator] node%d phase=%s state=%s finished=%d/%d elapsed=%.0fs' %
                      (rank,phase,'running' if local_code is None else 'waiting',len(results),nodes,now-started),
                      flush=True)
                next_heartbeat = now+heartbeat
            if now-started > timeout+2*stop_grace+120:
                raise TimeoutError('Missing preparation worker result after timeout')
            time.sleep(min(1., heartbeat))
        store.set(f'ack{rank}', '1')
        if rank == 0:
            store.wait([f'ack{i}' for i in range(nodes)], timedelta(seconds=120))
        return phase_code
    except BaseException:
        # local_exit=0 only means this node's child succeeded. A failed TCP
        # protocol is never a successful globally coordinated phase.
        phase_code = phase_code if phase_code else 1
        coordinator_error = traceback.format_exc()
        print(f'[control preparation coordinator] node{rank} phase={phase} coordination failed; '
              f'local_exit={local_code}, observed_results={results}. '
              'Check the earliest node0 error/server termination; local success does not confirm peer success.',flush=True)
        if not reported:
            try:
                store.set(keys[rank], '1')
            except Exception:
                pass
        raise
    finally:
        try:
            stop_child(child, stop_grace)
        except BaseException:
            phase_code = phase_code if phase_code else 1
            coordinator_error = coordinator_error or traceback.format_exc()
            raise
        finally:
            if phase_code is None: phase_code=1
            status('completed' if phase_code==0 else 'failed')
            print('[control preparation coordinator] node%d phase=%s exit=%s local_exit=%s all_results_observed=%s results=%s' %
                  (rank,phase,phase_code,local_code,confirmed,results),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=172800)
    parser.add_argument('--heartbeat', type=float, default=30)
    parser.add_argument('--phase', default='preparation')
    parser.add_argument('--status-file')
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
    print(f'[control preparation coordinator] node{rank} phase={args.phase} '
          f'TCPStore={os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]} nodes={nodes}',flush=True)
    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                          nodes, rank == 0, timedelta(seconds=600), use_libuv=False)
    raise SystemExit(run(command, store, rank, nodes, args.timeout, args.heartbeat,
                         phase=args.phase,status_file=args.status_file))


if __name__ == '__main__':
    main()
