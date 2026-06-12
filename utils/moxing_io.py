"""Small MoXing helpers for bounded on-demand OBS caching."""

import hashlib
import os
import shutil
import time
from pathlib import Path


REMOTE_PREFIXES = ("obs://", "s3://")


def is_remote_path(path):
    return isinstance(path, str) and path.startswith(REMOTE_PREFIXES)


def _mox():
    try:
        import moxing as mox
    except ImportError as exc:
        raise RuntimeError(
            "MoXing is required for OBS paths. Install the ModelArts moxing "
            "package in the Ascend environment.") from exc
    return mox


def _candidate_uris(path):
    yield path
    if path.startswith("obs://"):
        yield "s3://" + path[len("obs://"):]
    elif path.startswith("s3://"):
        yield "obs://" + path[len("s3://"):]


def remote_exists(path):
    last_error = None
    checked = False
    for candidate in _candidate_uris(path):
        try:
            exists = _mox().file.exists(candidate)
            checked = True
            if exists:
                return True
        except Exception as exc:
            last_error = exc
    if last_error is not None and not checked:
        raise last_error
    return False


def copy_file(source, destination):
    """Copy one file between local storage and OBS."""
    if not is_remote_path(source) and not is_remote_path(destination):
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        shutil.copy2(source, destination)
        return

    os.makedirs(
        os.path.dirname(os.path.abspath(destination)),
        exist_ok=True,
    ) if not is_remote_path(destination) else None
    last_error = None
    source_candidates = list(_candidate_uris(source)) if is_remote_path(
        source) else [source]
    destination_candidates = list(
        _candidate_uris(destination)) if is_remote_path(destination) else [
            destination]
    for source_candidate in source_candidates:
        for destination_candidate in destination_candidates:
            try:
                _mox().file.copy(source_candidate, destination_candidate)
                return
            except Exception as exc:
                last_error = exc
    raise RuntimeError(
        f"MoXing copy failed: {source} -> {destination}") from last_error


def copy_directory(source, destination):
    """Recursively synchronize a directory through MoXing."""
    if not is_remote_path(source) and not is_remote_path(destination):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    last_error = None
    source_candidates = list(_candidate_uris(source)) if is_remote_path(
        source) else [source]
    destination_candidates = list(
        _candidate_uris(destination)) if is_remote_path(destination) else [
            destination]
    for source_candidate in source_candidates:
        for destination_candidate in destination_candidates:
            try:
                _mox().file.copy_parallel(
                    source_candidate, destination_candidate)
                return
            except Exception as exc:
                last_error = exc
    raise RuntimeError(
        f"MoXing directory copy failed: {source} -> {destination}"
    ) from last_error


def read_text(path):
    if not is_remote_path(path):
        with open(path) as handle:
            return handle.read()
    last_error = None
    for candidate in _candidate_uris(path):
        try:
            payload = _mox().file.read(candidate, binary=False)
            return payload.decode("utf-8") if isinstance(
                payload, bytes) else payload
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read OBS text file: {path}") from last_error


def write_text(path, content):
    if not is_remote_path(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(content)
        return
    last_error = None
    for candidate in _candidate_uris(path):
        try:
            _mox().file.write(candidate, content)
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to write OBS text file: {path}") from last_error


def join_remote(root, relative):
    return root.rstrip("/") + "/" + relative.lstrip("/")


def _cache_path(remote_path, cache_dir):
    digest = hashlib.sha256(remote_path.encode("utf-8")).hexdigest()
    suffix = Path(remote_path.split("?", 1)[0]).suffix
    basename = os.path.basename(remote_path.split("?", 1)[0])
    safe_name = basename if basename else f"object{suffix}"
    return os.path.join(cache_dir, digest[:2], f"{digest}-{safe_name}")


def _acquire_lock(lock_path, timeout, stale_seconds):
    deadline = time.time() + timeout
    while True:
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            return True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_seconds:
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                continue
            if time.time() >= deadline:
                return False
            time.sleep(0.25)


def prune_cache(cache_dir, max_bytes, protected=()):
    if max_bytes <= 0 or not os.path.isdir(cache_dir):
        return
    files = []
    total = 0
    protected = {os.path.abspath(path) for path in protected}
    for root, _, names in os.walk(cache_dir):
        for name in names:
            if name.endswith((".lock", ".partial")):
                continue
            path = os.path.join(root, name)
            if os.path.abspath(path) in protected:
                continue
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            total += stat.st_size
            files.append((stat.st_atime, stat.st_size, path))
    if total <= max_bytes:
        return
    for _, size, path in sorted(files):
        try:
            os.unlink(path)
            total -= size
        except FileNotFoundError:
            pass
        if total <= max_bytes:
            break


def stage_remote_file(remote_path, cache_dir, max_cache_bytes=0,
                      retries=3, lock_timeout=900):
    """Materialize one OBS object as a seekable local file."""
    if not is_remote_path(remote_path):
        return remote_path
    local_path = _cache_path(remote_path, cache_dir)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        os.utime(local_path, None)
        return local_path

    lock_path = local_path + ".lock"
    owns_lock = _acquire_lock(
        lock_path, timeout=lock_timeout, stale_seconds=lock_timeout * 2)
    if not owns_lock:
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        raise TimeoutError(f"Timed out waiting for OBS cache: {remote_path}")

    partial_path = local_path + f".{os.getpid()}.partial"
    try:
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        last_error = None
        for attempt in range(retries):
            try:
                copy_file(remote_path, partial_path)
                if os.path.getsize(partial_path) <= 0:
                    raise IOError("downloaded file is empty")
                os.replace(partial_path, local_path)
                os.utime(local_path, None)
                prune_cache(
                    cache_dir, max_cache_bytes, protected=(local_path,))
                return local_path
            except Exception as exc:
                last_error = exc
                if os.path.exists(partial_path):
                    os.unlink(partial_path)
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Failed to stage OBS file after {retries} attempts: "
            f"{remote_path}") from last_error
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
