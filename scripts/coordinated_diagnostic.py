"""Keep all ModelArts nodes alive while node zero runs a serial diagnostic."""
import argparse
from datetime import timedelta
import os
import subprocess
import time

import torch.distributed as dist


def run(command, store, rank, nodes, timeout, heartbeat=30):
    started = time.monotonic()
    child = None
    code = 1
    try:
        if rank == 0:
            try:
                child = subprocess.Popen(command)
                while child.poll() is None:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError('diagnostic execution timeout')
                    print('[diagnostic coordinator] leader running; elapsed %.0fs' %
                          (time.monotonic() - started), flush=True)
                    time.sleep(heartbeat)
                code = child.returncode
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                store.set('result', str(code))
        else:
            while not store.check(['result']):
                if time.monotonic() - started > timeout + 120:
                    raise TimeoutError('waiting for leader timed out')
                print('[diagnostic coordinator] node%d waiting for leader; elapsed %.0fs' %
                      (rank, time.monotonic() - started), flush=True)
                time.sleep(heartbeat)
            code = int(store.get('result').decode())
        store.set('ack%d' % rank, '1')
        if rank == 0:
            store.wait(['ack%d' % i for i in range(nodes)], timedelta(seconds=120))
        return code if 0 <= code <= 255 else 1
    finally:
        print('[diagnostic coordinator] node%d exit code=%d' % (rank, code), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('missing command')
    rank, nodes = int(os.environ['NODE_RANK']), int(os.environ['NNODES'])
    timeout = int(os.environ.get('AUDIT_TIMEOUT_SECONDS', '7200'))
    # TCPStore only: no HCCL process group, no NPU memory on waiting nodes.
    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                          nodes, rank == 0, timedelta(seconds=180), use_libuv=False)
    raise SystemExit(run(args.command, store, rank, nodes, timeout))


if __name__ == '__main__':
    main()
