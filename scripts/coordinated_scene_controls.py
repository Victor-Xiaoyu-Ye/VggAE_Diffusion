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


def write_status(status_file, value):
    if not status_file:
        return
    try:
        path = Path(status_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
        os.replace(temp, path)
    except OSError as exc:
        # Logging cannot tear down the owner or hide the original failure.
        print(f'[control preparation coordinator] status write warning: {exc!r}', flush=True)


def rendezvous(store, rank, nodes, timeout, heartbeat=30, phase='preparation',
               run_id='', status_file=None):
    """Register named nodes before ANY child starts; report missing/duplicate ranks.

    TCPStore's implicit worker wait blocks only the owner. Replacing it with an
    explicit registration phase also lets the owner stay alive while connected
    peers read a startup failure, rather than dropping them with recv 0 bytes.
    """
    started = time.monotonic()
    next_heartbeat = started
    peers = {}
    error_key = 'join_error'
    record = dict(node=rank, nodes=nodes, phase=phase, run_id=run_id,
                  hostname=socket.gethostname(), pid=os.getpid(),
                  vc_task_index=os.environ.get('VC_TASK_INDEX'))
    payload = json.dumps(record, sort_keys=True)

    def status(state, error=None):
        write_status(status_file, dict(
            **record, state=state, local_exit=None,
            phase_exit=1 if error else None, all_results_observed=False, results={},
            master_addr=os.environ.get('MASTER_ADDR'), master_port=os.environ.get('MASTER_PORT'),
            joined_nodes=peers, missing_nodes=[i for i in range(nodes) if i not in peers],
            coordinator_error=error, elapsed_seconds=time.monotonic()-started, time=time.time()))

    def fail(message):
        # Keep the first cause, including which node/phase registered incorrectly.
        store.compare_set(error_key, '', message)

    try:
        claimed = store.compare_set(f'join{rank}', '', payload).decode()
        if claimed != payload:
            fail(f'Duplicate NODE_RANK={rank}: registered={claimed}, incoming={payload}')
        while True:
            for i in range(nodes):
                if i not in peers and store.check([f'join{i}']):
                    peers[i] = json.loads(store.get(f'join{i}').decode())
                    expected = dict(node=i, nodes=nodes, phase=phase, run_id=run_id)
                    if any(peers[i].get(key) != value for key, value in expected.items()):
                        fail(f'Rendezvous identity mismatch: node{rank} expects {expected}, '
                             f'node{i} registered {peers[i]}')
            missing = [i for i in range(nodes) if i not in peers]
            now = time.monotonic()
            if missing and now-started >= timeout:
                fail(f'Rendezvous timed out after {timeout:g}s: joined={sorted(peers)}, '
                     f'missing={missing}; check the missing node startup, NODE_RANK, '
                     'NNODES, MASTER_ADDR and MASTER_PORT.')
            if store.check([error_key]):
                reason = store.get(error_key).decode()
                status('join_failed', reason)
                store.set(f'join_failed_ack{rank}', '1')
                if rank == 0:
                    # Missing nodes cannot ack. Wait only for registered peers;
                    # no child has been started and their original error is saved.
                    try:
                        store.wait([f'join_failed_ack{i}' for i in peers], timedelta(seconds=30))
                    except Exception as exc:
                        print(f'[control preparation coordinator] startup failure ack warning: {exc}', flush=True)
                raise RuntimeError(reason)
            if not missing:
                status('joined')
                print(f'[control preparation coordinator] node{rank} phase={phase} '
                      f'run_id={run_id} joined={sorted(peers)}; starting child', flush=True)
                return peers
            if now >= next_heartbeat:
                print(f'[control preparation coordinator] node{rank} phase={phase} '
                      f'run_id={run_id} state=joining joined={sorted(peers)} missing={missing} '
                      f'elapsed={now-started:.0f}s join_timeout={timeout:g}s', flush=True)
                status('joining')
                next_heartbeat = now+heartbeat
            time.sleep(min(1., heartbeat))
    except BaseException:
        status('join_failed', traceback.format_exc())
        print(f'[control preparation coordinator] node{rank} phase={phase} startup failed; '
              f'joined={sorted(peers)} missing={[i for i in range(nodes) if i not in peers]}; '
              'child not started', flush=True)
        raise


def connect_and_join(rank, nodes, timeout, heartbeat, phase, run_id, status_file):
    address, port = os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])
    initial = dict(node=rank, nodes=nodes, phase=phase, run_id=run_id, state='connecting',
                   hostname=socket.gethostname(), pid=os.getpid(),
                   master_addr=address, master_port=port, time=time.time(),
                   local_exit=None, phase_exit=None, all_results_observed=False, results={})
    write_status(status_file, initial)
    print(f'[control preparation coordinator] node{rank} phase={phase} run_id={run_id} '
          f'TCPStore={address}:{port} nodes={nodes} hostname={initial["hostname"]} '
          f'VC_TASK_INDEX={os.environ.get("VC_TASK_INDEX")} join_timeout={timeout:g}s', flush=True)
    try:
        store = dist.TCPStore(address, port, nodes, rank == 0, timedelta(seconds=timeout),
                              wait_for_workers=False, use_libuv=False)
    except BaseException:
        write_status(status_file, dict(initial, state='connect_failed', phase_exit=1,
                                       coordinator_error=traceback.format_exc()))
        raise
    # Connection establishment can be slow; connected operations should not
    # block for the entire join budget if the server becomes unreachable.
    store.set_timeout(timedelta(seconds=min(60., timeout)))
    peers = rendezvous(store, rank, nodes, timeout, heartbeat, phase, run_id, status_file)
    return store, peers


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
        phase='preparation', status_file=None, joined_nodes=None, run_id=''):
    started = time.monotonic()
    child = None
    child_pid = None
    child_returncode = None
    child_signal = None
    termination_reason = None
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
    for flag in ('--stage','--phase','--arm','--stop-after'):
        if flag in command and command.index(flag)+1<len(command):
            job[flag[2:]]=str(command[command.index(flag)+1])

    def status(state):
        value=dict(phase=phase,run_id=run_id,joined_nodes=joined_nodes,
            job=job,node=rank,state=state,local_exit=local_code,
            child_pid=child_pid,child_returncode=child_returncode,
            child_signal=child_signal,termination_reason=termination_reason,
            phase_exit=phase_code,all_results_observed=confirmed,results=results,
            local_error=local_error,coordinator_error=coordinator_error,
            hostname=socket.gethostname(),pid=os.getpid(),
            master_addr=os.environ.get('MASTER_ADDR'),master_port=os.environ.get('MASTER_PORT'),
            elapsed_seconds=time.monotonic()-started,time=time.time())
        write_status(status_file, value)

    def record_child_exit(code):
        nonlocal child_returncode, child_signal, termination_reason
        if code is None or child_returncode is not None:
            return
        child_returncode = code
        # Popen uses negative signal numbers on POSIX. Keep large Windows
        # exception/status codes verbatim, without mislabeling them as signals.
        if -128 < code < 0:
            try:
                child_signal = signal.Signals(-code).name
            except ValueError:
                child_signal = f'SIGNAL_{-code}'
        if termination_reason is None:
            termination_reason = 'signal' if child_signal else 'exited'
        if code:
            print(f'[control preparation coordinator] node{rank} phase={phase} '
                  f'child_pid={child_pid} child_returncode={code} '
                  f'child_signal={child_signal} reason={termination_reason} '
                  f'job={job}', flush=True)

    try:
        status('starting')
        try:
            child = subprocess.Popen(command, start_new_session=os.name == 'posix')
            pid = getattr(child, 'pid', None)
            child_pid = pid if isinstance(pid, int) else None
        except Exception:
            # In particular, the node0 TCPStore owner must NOT unwind here.
            # Publish the launch failure through the normal result/ack protocol
            # so every live peer can read it before the server is destroyed.
            local_code = 1
            termination_reason = 'launch_failure'
            local_error = traceback.format_exc()
            print(f'[control preparation coordinator] node{rank} phase={phase} child launch failed:\n{local_error}',flush=True)
        while True:
            now = time.monotonic()
            if local_code is None:
                child_code = child.poll()
                if child_code is not None:
                    record_child_exit(child_code)
                    local_code = child_code if 0 <= child_code <= 255 else 1
                elif now-started > timeout:
                    print('[control preparation coordinator] command timed out', flush=True)
                    local_code = 124
                    termination_reason = 'timeout'
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
                termination_reason = 'peer_failure'
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
            if child is not None and child.poll() is None and termination_reason is None:
                termination_reason = 'coordinator_cleanup'
            stop_child(child, stop_grace)
            if child is not None:
                record_child_exit(child.poll())
        except BaseException:
            phase_code = phase_code if phase_code else 1
            coordinator_error = coordinator_error or traceback.format_exc()
            raise
        finally:
            if phase_code is None: phase_code=1
            status('completed' if phase_code==0 else 'failed')
            print('[control preparation coordinator] node%d phase=%s exit=%s local_exit=%s all_results_observed=%s results=%s child_pid=%s child_returncode=%s child_signal=%s' %
                  (rank,phase,phase_code,local_code,confirmed,results,
                   child_pid,child_returncode,child_signal),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=172800)
    parser.add_argument('--heartbeat', type=float, default=30)
    parser.add_argument('--join-timeout', type=float, default=600,
                        help='Seconds for connecting/registration, independent of child runtime')
    parser.add_argument('--phase', default='preparation')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--status-file')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or min(args.timeout, args.heartbeat, args.join_timeout) <= 0:
        parser.error('a command and positive timeout/heartbeat/join-timeout are required')
    def terminate(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    rank, nodes = int(os.environ['NODE_RANK']), int(os.environ['NNODES'])
    if nodes < 1 or not 0 <= rank < nodes:
        parser.error(f'Invalid NODE_RANK={rank} for NNODES={nodes}')
    store, peers = connect_and_join(rank, nodes, args.join_timeout, args.heartbeat,
                                    args.phase, args.run_id, args.status_file)
    raise SystemExit(run(command, store, rank, nodes, args.timeout, args.heartbeat,
                         phase=args.phase, status_file=args.status_file,
                         joined_nodes=peers, run_id=args.run_id))


if __name__ == '__main__':
    main()
