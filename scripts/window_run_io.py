#!/usr/bin/env python3
"""Incremental two-destination snapshots and fail-visible ModelArts staging."""
import argparse
import json
import os
import signal
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.moxing_io import copy_file, remote_exists, join_remote, is_remote_path
from utils.output_snapshot import build_snapshot, cleanup_snapshot


def child(root, name):
    return join_remote(root, name) if is_remote_path(root) else str(Path(root)/name)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def publish(source, roots):
    """One publisher per source. Record success per file/destination after copy.

    Snapshots pin checkpoint inodes; JSONL snapshots stop at the observed EOF.
    The ledger makes periodic sync incremental and final sync cheap. A failed
    destination never prevents attempting the other destination.
    """
    source = Path(source)
    ledger_path = source/'.window_sync_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    snapshot, _ = build_snapshot(source)
    errors = []
    try:
        for root in dict.fromkeys(roots):
            saved = ledger.setdefault(root, {})
            try:
                for file in sorted(Path(snapshot).rglob('*')):
                    if not file.is_file() or file.name in ('.window_sync_ledger.json', 'publication_status.json'):
                        continue
                    name = file.relative_to(snapshot).as_posix()
                    stat = file.stat(); stamp = [stat.st_size, stat.st_mtime_ns]
                    if saved.get(name) == stamp: continue
                    copy_file(str(file), child(root, name))
                    saved[name] = stamp
                atomic(ledger_path, ledger)
            except Exception as exc:
                errors.append({'destination':root,'error':repr(exc)})
                atomic(ledger_path, ledger)
        receipt = dict(schema='window-publication-v1', status='failed' if errors else 'synced',
                       destinations=list(dict.fromkeys(roots)), errors=errors, unix_time=time.time())
        atomic(source/'publication_status.json', receipt)
        for root in dict.fromkeys(roots):
            try: copy_file(str(source/'publication_status.json'), child(root,'publication_status.json'))
            except Exception as exc: errors.append({'destination':root,'error':repr(exc)})
        if errors:
            atomic(source/'publication_status.json',dict(receipt,status='failed',errors=errors))
            raise RuntimeError(f'publication failed: {errors}')
    finally:
        cleanup_snapshot(snapshot)


def stage(destination, candidates):
    """Download into a new partial file, then rename. Retry the second root."""
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    for source in dict.fromkeys(candidates):
        if not source: continue
        partial = destination.with_name(destination.name+'.partial')
        try:
            copy_file(source, str(partial))
            if not partial.stat().st_size: raise ValueError('empty staged artifact')
            os.replace(partial,destination)
            print(f'Staged from {source}', flush=True)
            return source
        except Exception as exc:
            print(f'[read fallback] {source}: {exc}', file=sys.stderr, flush=True)
            partial.unlink(missing_ok=True)
    raise FileNotFoundError(f'no readable input among {candidates}')


def main():
    def stop(signum, frame):
        raise SystemExit(0)  # Run snapshot cleanup before the launcher final sync.
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('guard','stage','publish'))
    p.add_argument('--source',default='')
    p.add_argument('--destination',default='')
    p.add_argument('--root',action='append',default=[])
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=int,default=60)
    a=p.parse_args()
    if not a.root: p.error('at least one --root required')
    if a.mode == 'guard':
        for root in dict.fromkeys(a.root):
            if is_remote_path(root):
                import moxing as mox
                exists = mox.file.exists(root)  # Permission/network failures must not mean "fresh".
            else:
                exists = Path(root).exists()
            if exists:
                raise FileExistsError(f'Existing output {root}; use a fresh WINDOW_NAMESPACE or explicit RESUME=1')
    elif a.mode == 'stage':
        stage(a.destination,a.root)
    else:
        while True:
            try: publish(a.source,a.root)
            except Exception as exc:
                if not a.watch: raise
                print('[window-sync] '+repr(exc),flush=True)
            if not a.watch: break
            time.sleep(max(10,a.interval))


if __name__ == '__main__': main()
