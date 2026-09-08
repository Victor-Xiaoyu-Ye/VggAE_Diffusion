"""Atomic, torch-free run completion records and shell-side verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class FlowRunStatus:
    def __init__(self, output_dir, args):
        self.path = Path(output_dir) / 'run_status.json'
        self.payload = {'schema': 'r7-flow-run-status-v1', 'status': 'running',
            'phase': 'initializing', 'step': 0, 'expected_steps': args.max_steps,
            'stop_after_steps': args.stop_after_steps, 'pid': os.getpid(),
            'prediction': args.prediction, 'noise_mode': args.noise_mode,
            'args': vars(args)}

    def update(self, **values):
        self.payload.update(values)
        self.payload['updated_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, self.payload)
        print('[flow-status] ' + json.dumps({k: self.payload[k] for k in
            ('status', 'phase', 'step', 'expected_steps')}, ensure_ascii=False), flush=True)


def verify_run(output_dir, expected_steps, stop_after_steps=0):
    directory = Path(output_dir)
    status = json.loads((directory / 'run_status.json').read_text(encoding='utf-8'))
    stop = min(expected_steps, stop_after_steps) if stop_after_steps > 0 else expected_steps
    expected_status = 'completed' if stop == expected_steps else 'paused'
    checkpoint = 'checkpoint_final.pt' if expected_status == 'completed' else 'checkpoint_latest.pt'
    if (status.get('schema') != 'r7-flow-run-status-v1'
            or status.get('status') != expected_status or status.get('step') != stop
            or status.get('expected_steps') != expected_steps
            or status.get('checkpoint') != checkpoint):
        raise ValueError(f'trainer did not finish requested budget: {status}')
    artifact = directory / checkpoint
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError(f'missing final committed artifact: {artifact}')
    return status
