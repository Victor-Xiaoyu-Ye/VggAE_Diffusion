"""Bounded stage budgets and content-checked dual artifact publication/recovery."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch
from scripts.window_run_io import child
from utils.moxing_io import copy_file, read_text
from utils.native_audit_io import exists, sha256, safe_relative
from utils.training import atomic_torch_save


def atomic(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    for attempt in range(10):
        try:
            os.replace(temp,path)
            break
        except PermissionError:
            # Windows cannot replace an open destination; another local rank
            # may be reading this tiny receipt. Linux rename is unaffected.
            if os.name!='nt' or attempt==9:
                raise
            time.sleep(.01*(attempt+1))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StageBudget:
    def __init__(self, seconds, spent=0.):
        self.seconds, self.spent = seconds, spent
        self.start = time.monotonic()
        self.interrupted = False

    def elapsed(self):
        return self.spent + time.monotonic()-self.start

    def due(self):
        return self.elapsed() >= self.seconds or self.interrupted

    def install_signals(self):
        def stop(*_):
            self.interrupted = True
            print('[stage] termination requested; checkpoint at next optimizer/shard boundary', flush=True)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)


class ArtifactStore:
    def __init__(self, local, roots):
        self.local = Path(local)
        self.local.mkdir(parents=True, exist_ok=True)
        self.roots = list(dict.fromkeys(r for r in roots if r))
        self.status_name = f'publication_status-r{int(os.environ.get("RANK",0)):03d}.json'
        self.verified = {}

    def json(self, name, required=False):
        errors, candidates = [], []
        for root in [str(self.local), *self.roots]:
            try:
                path = child(root, safe_relative(name))
                if exists(path):
                    value=json.loads(read_text(path))
                    if not name.endswith('.pt.json'):
                        return value
                    candidates.append(value)
            except Exception as exc:
                errors.append(repr(exc))
        if candidates:
            return max(candidates,key=lambda x:x.get('committed_unix_ns',0))
        if required or errors:
            raise RuntimeError(f'Cannot recover {name}: {errors}')
        return None

    def publish(self, names):
        """Try both destinations. At least one complete durable replica required.

        Caller orders payload before receipt. Failures are recorded, never
        mistaken for a successful dual write. A later save retries both roots.
        """
        errors, success = [], []
        for root in self.roots:
            try:
                for name in names:
                    copy_file(str(self.local/safe_relative(name)), child(root, name))
                success.append(root)
            except Exception as exc:
                errors.append(dict(root=root, error=repr(exc)))
        atomic(self.local/self.status_name, dict(success=success, errors=errors, time=time.time()))
        for root in self.roots:
            try:
                copy_file(str(self.local/self.status_name),child(root,self.status_name))
            except Exception:
                pass  # The payload errors above remain the authoritative result.
        if errors:
            print('[dual-write degraded] '+json.dumps(errors), flush=True)
        if self.roots and not success:
            raise RuntimeError('No durable output destination accepted this commit')

    def write_json(self, name, value):
        atomic(self.local/safe_relative(name), value)
        self.publish([name])

    def save(self, name, payload, identity):
        name = safe_relative(name)
        path = self.local/name
        atomic_torch_save(payload, str(path))
        receipt = dict(file=name, size=path.stat().st_size, sha256=sha256(path), identity=identity,
                       committed_unix_ns=time.time_ns())
        atomic(self.local/(name+'.json'), receipt)
        self.publish([name, name+'.json'])
        return receipt

    def load(self, name, identity=None, required=False):
        receipt = self.json(name+'.json', required)
        if receipt is None:
            return None
        if identity is not None and receipt['identity'] != identity:
            raise ValueError('Artifact identity mismatch: '+name)
        path = self.local/safe_relative(name)
        def valid(p):
            if not p.is_file() or p.stat().st_size != receipt['size']:
                return False
            signature=(p.stat().st_size,p.stat().st_mtime_ns,receipt['sha256'])
            if self.verified.get(str(p))==signature:
                return True
            if sha256(p)!=receipt['sha256']:
                return False
            self.verified[str(p)]=signature
            return True
        if not valid(path):
            errors = []
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name+f'.{os.getpid()}.partial')
            for root in self.roots:
                try:
                    copy_file(child(root, name), str(partial))
                    if not valid(partial):
                        raise ValueError('Invalid payload checksum')
                    os.replace(partial, path)
                    break
                except Exception as exc:
                    errors.append(repr(exc))
                    partial.unlink(missing_ok=True)
            else:
                raise RuntimeError(f'Unrecoverable committed artifact {name}: {errors}')
        atomic(self.local/(name+'.json'),receipt)
        # torch_npu's mmap loader requires str, unlike stock PyTorch's Path support.
        return torch.load(str(path), map_location='cpu', weights_only=False, mmap=os.name!='nt')
