"""Freeze and stage one shared initialization/cohort for matched camera arms."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.window_run_io import atomic, child, stage
from utils.moxing_io import copy_file, is_remote_path, read_text


def exists(path):
    # Network/permission errors must never be mistaken for a fresh experiment.
    if is_remote_path(path):
        import moxing
        return bool(moxing.file.exists(path))
    return Path(path).exists()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def receipts(output, roots, name):
    found = []
    for source in [str(output/name), *(child(root, name) for root in roots)]:
        if exists(source):
            found.append((source, json.loads(read_text(source))))
    if found and any(value != found[0][1] for _, value in found[1:]):
        raise ValueError(f'Conflicting immutable {name}: {[source for source, _ in found]}')
    return found[0][1] if found else None


def checked_stage(output, name, expected_hash, roots):
    destination = output/name
    if destination.is_file() and sha256(destination) == expected_hash:
        return
    errors = []
    for root in roots:
        source = child(root, name)
        partial = destination.with_name(destination.name+'.partial')
        try:
            copy_file(source, str(partial))
            if sha256(partial) != expected_hash:
                raise ValueError('SHA256 mismatch')
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, destination)
            print(f'[camera inputs] verified {source}', flush=True)
            return
        except Exception as exc:
            partial.unlink(missing_ok=True)
            errors.append(dict(source=source, error=repr(exc)))
    raise RuntimeError(f'No valid copy of {name}: {errors}')


def freeze(output, roots, sources):
    receipt = receipts(output, roots, 'initialization.json')
    if receipt is None:
        if not sources:
            raise ValueError('Freezing initialization requires source checkpoint roots')
        import torch
        source_file = output/'source_checkpoint.pt'
        source = stage(source_file, sources)
        source_hash = sha256(source_file)
        saved = torch.load(str(source_file), map_location='cpu', weights_only=False, mmap=True)
        required = ('schema', 'step', 'model_args', 'statistics', 'contract', 'ema')
        if saved.get('schema') != 'r7-window-trainer-v1' or saved.get('step', 0) < 1:
            raise ValueError('Source checkpoint is not a trained window DiT')
        expected_model = dict(channels=192, grid=18, future=4, temporal_factor=2,
                              width=1536, depth=24, heads=24, text_dim=4096)
        if saved.get('model_args') != expected_model:
            raise ValueError('Camera pilot requires the full-HQ 1.65B source architecture')
        from utils.window_codec import validate_runtime
        if validate_runtime(saved['statistics']['representation']) != 'legacy':
            raise ValueError('Camera pilot source must use frozen legacy R7 statistics')
        compact = {key: saved[key] for key in required}
        compact['initialization_only'] = True
        partial = output/'initialization_ema.pt.partial'
        torch.save(compact, partial)
        frozen_hash = sha256(partial)
        filename = f'initialization_ema_{frozen_hash}.pt'
        os.replace(partial, output/filename)
        receipt = dict(schema='r7-camera-initialization-v1', file=filename,
                       sha256=frozen_hash, step=int(saved['step']), weights='ema',
                       source=source, source_sha256=source_hash,
                       size=(output/filename).stat().st_size,
                       note='EMA-only initialization; optimizer and counters reset in each arm')
        # The compact artifact retains all trainer identity checks. Drop the
        # temporary optimizer-bearing source after serializing shared tensors.
        del compact, saved
        source_file.unlink()
        atomic(output/'initialization.json', receipt)
    else:
        if receipt.get('schema') != 'r7-camera-initialization-v1':
            raise ValueError('Unsupported initialization receipt')
        checked_stage(output, receipt['file'], receipt['sha256'], roots)
        atomic(output/'initialization.json', receipt)
    errors = []
    for root in roots:
        try:
            remote_receipt = child(root, 'initialization.json')
            if exists(remote_receipt) and json.loads(read_text(remote_receipt)) == receipt:
                if exists(child(root, receipt['file'])):
                    continue
            copy_file(str(output/receipt['file']), child(root, receipt['file']))
            copy_file(str(output/'initialization.json'), remote_receipt)
        except Exception as exc:
            errors.append(dict(root=root, error=repr(exc)))
    if errors:
        raise RuntimeError(f'Initialization double publication failed: {errors}')
    print(f"[camera inputs] shared EMA step={receipt['step']} SHA256={receipt['sha256']}", flush=True)


def publish_inputs(output, roots, names):
    initial = receipts(output, roots, 'initialization.json')
    if initial is None:
        raise ValueError('Freeze initialization before publishing a cohort')
    names = sorted(set([*names, 'initialization.json', initial['file']]))
    files = {}
    for name in names:
        if Path(name).name != name or not (output/name).is_file():
            raise ValueError(f'Missing/unsafe immutable input: {name}')
        files[name] = dict(sha256=sha256(output/name), size=(output/name).stat().st_size)
    receipt = dict(schema='r7-camera-inputs-v1', files=files,
                   initialization_sha256=initial['sha256'], initialization_step=initial['step'])
    old = receipts(output, roots, 'inputs.json')
    if old is not None and old != receipt:
        raise ValueError('Refusing to replace frozen camera input cohort')
    errors = []
    for root in roots:
        try:
            committed = child(root, 'inputs.json')
            if (exists(committed) and json.loads(read_text(committed)) == receipt and
                    all(exists(child(root, name)) for name in names)):
                continue
            # Avoid resending a multi-GiB snapshot only when its separate commit
            # receipt proves that this root already completed that publication.
            initialization_receipt = child(root, 'initialization.json')
            initial_committed = (exists(initialization_receipt) and
                json.loads(read_text(initialization_receipt)) == initial and
                exists(child(root, initial['file'])))
            if not initial_committed:
                copy_file(str(output/initial['file']), child(root, initial['file']))
            for name in names:
                if name == initial['file']:
                    continue
                copy_file(str(output/name), child(root, name))
            atomic(output/'inputs.json', receipt)
            copy_file(str(output/'inputs.json'), child(root, 'inputs.json'))
        except Exception as exc:
            errors.append(dict(root=root, error=repr(exc)))
    if errors:
        raise RuntimeError(f'Camera input double publication failed: {errors}')


def fetch_inputs(output, roots, allow_missing=False):
    receipt = receipts(output, roots, 'inputs.json')
    if receipt is None:
        if allow_missing:
            return 3
        raise FileNotFoundError('No committed camera inputs; run preparation first')
    if receipt.get('schema') != 'r7-camera-inputs-v1':
        raise ValueError('Unsupported camera input receipt')
    for name, value in receipt['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe input receipt filename')
        checked_stage(output, name, value['sha256'], roots)
    atomic(output/'inputs.json', receipt)
    if allow_missing:
        # This option is used by the leader's prepare-or-reuse path. A failed
        # second write may have left a valid local/first-root commit: repair all
        # destinations before telling the launcher that preparation is complete.
        # Worker fetches are read-only and never race to republish shared inputs.
        publish_inputs(output, roots, list(receipt['files']))
    print(f"[camera inputs] verified frozen cohort + EMA step={receipt['initialization_step']}", flush=True)
    return 0


def distributed_fetch(args):
    """Join TCP before staging, then heartbeat until all nodes finish their copy."""
    import torch.distributed as dist
    rank, nodes = int(os.environ['NODE_RANK']), int(os.environ['NNODES'])
    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                          nodes, rank == 0, timedelta(seconds=180), use_libuv=False)
    command = [sys.executable, __file__, 'fetch', '--output', str(args.output)]
    for root in args.root:
        command.extend(('--root', root))
    worker = subprocess.Popen(command)
    started = time.monotonic()
    try:
        while worker.poll() is None:
            if time.monotonic()-started > args.timeout:
                raise TimeoutError('Camera input staging timeout')
            print(f'[camera input coordinator] node{rank} staging; elapsed {time.monotonic()-started:.0f}s', flush=True)
            time.sleep(30)
        code = worker.returncode
        store.set(f'result{rank}', str(code))
        keys = [f'result{index}' for index in range(nodes)]
        while not store.check(keys):
            if time.monotonic()-started > args.timeout+120:
                raise TimeoutError('Waiting for other camera input stages timed out')
            print(f'[camera input coordinator] node{rank} waiting for all nodes', flush=True)
            time.sleep(30)
        codes = [int(store.get(key).decode()) for key in keys]
        store.set(f'ack{rank}', '1')
        if rank == 0:
            store.wait([f'ack{index}' for index in range(nodes)], timedelta(seconds=120))
        if any(codes):
            raise RuntimeError(f'Camera input staging failed: node exit codes {codes}')
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'publish', 'fetch', 'fetch_all'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--file', action='append', default=[])
    parser.add_argument('--allow_missing', action='store_true')
    parser.add_argument('--timeout', type=int, default=172800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.root = list(dict.fromkeys(args.root))
    if args.mode == 'freeze':
        freeze(args.output, args.root, args.source)
    elif args.mode == 'publish':
        publish_inputs(args.output, args.root, args.file)
    elif args.mode == 'fetch_all':
        distributed_fetch(args)
    else:
        raise SystemExit(fetch_inputs(args.output, args.root, args.allow_missing))


if __name__ == '__main__':
    main()
