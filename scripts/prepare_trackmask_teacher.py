#!/usr/bin/env python3
"""Prepare the pinned official full-head VGGT checkpoint at an explicit path.

The stage55 launcher runs this helper only with the explicit
VGGT_TRACKMASK_DOWNLOAD_OFFICIAL=1 opt-in. Use a cluster /cache destination;
--inspect-only reports provenance without downloading.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.request


REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'
FILENAME = 'model.safetensors'
SIZE = 5026367224
SHA256 = 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
URL = f'https://huggingface.co/facebook/VGGT-1B/resolve/{REVISION}/{FILENAME}'


def emit_progress(value):
    print('[trackmask teacher preparation] '+json.dumps(value, sort_keys=True), flush=True)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path, size, sha256):
    path = Path(path)
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == sha256


def prepare(destination, *, url=URL, size=SIZE, sha256=SHA256, attempts=3,
            timeout=30, progress_seconds=15, chunk_size=8 * 1024 * 1024,
            emit=emit_progress, sleep=time.sleep):
    """Resume only verified HTTP byte ranges; promote only complete hash matches.

    A failed download never modifies an existing final file. A partial file is
    retained after network failures, but removed after a full checksum failure.
    """
    destination = Path(destination).expanduser().resolve()
    if attempts < 1 or timeout <= 0 or progress_seconds <= 0 or chunk_size < 1:
        raise ValueError('attempts, timeout, progress_seconds and chunk_size must be positive')
    if size <= 0 or not re.fullmatch(r'[0-9a-f]{64}', sha256):
        raise ValueError('a positive expected size and lowercase SHA256 are required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def report(phase, byte_count=0, **extra):
        emit(dict(phase=phase, bytes=int(byte_count), expected_bytes=size,
                  elapsed_seconds=time.monotonic()-started, DI_throughput=0.0,
                  DI_note='network/checksum work; no useful NPU inference', **extra))

    def receipt(status):
        value = dict(schema='vggt-trackmask-teacher-download-v1', status=status,
                     source=url, destination=str(destination), size=size, sha256=sha256,
                     revision=REVISION if url == URL else None, unix_time=time.time())
        path = destination.with_name(destination.name+'.receipt.json')
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
        os.replace(temporary, path)
        return value

    if destination.exists():
        report('verify_existing', destination.stat().st_size)
        if is_verified(destination, size, sha256):
            report('already_verified', size)
            return receipt('already_verified')
        report('existing_file_mismatch', destination.stat().st_size,
               note='Existing final file retained until verified replacement is ready.')
    partial = destination.with_name(destination.name+'.partial')
    last_error = None
    for attempt in range(1, attempts+1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= size:
                if offset == size and is_verified(partial, size, sha256):
                    os.replace(partial, destination)
                    report('completed_from_partial', size, attempt=attempt)
                    return receipt('downloaded')
                partial.unlink()
                offset = 0
            headers = {'User-Agent': 'VGGAE-trackmask-teacher/1', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            report('connecting', offset, attempt=attempt)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match:
                        raise ValueError(f'Invalid partial response Content-Range: {content_range!r}')
                    begin, end, total = map(int, match.groups())
                    if begin != offset or total != size or not begin <= end < total:
                        raise ValueError(f'Unexpected partial range: {content_range!r}; offset={offset}, total={size}')
                    expected_response_bytes = end-begin+1
                elif status == 200:
                    if offset:
                        report('range_ignored_restart', offset, attempt=attempt)
                    offset = 0
                    expected_response_bytes = size
                else:
                    raise ValueError(f'Unexpected HTTP status {status}')
                content_length = response.headers.get('Content-Length')
                if content_length is not None and int(content_length) != expected_response_bytes:
                    raise ValueError(f'Unexpected Content-Length {content_length}; expected {expected_response_bytes}')
                mode = 'ab' if offset else 'wb'
                received = 0
                last_progress = time.monotonic()
                with partial.open(mode) as stream:
                    while True:
                        # read1 returns available bytes instead of waiting to
                        # fill a large buffer on a slow but active connection.
                        chunk = response.read1(chunk_size)
                        if not chunk:
                            break
                        if received+len(chunk) > expected_response_bytes or offset+received+len(chunk) > size:
                            raise ValueError('HTTP response exceeds its declared/expected byte range')
                        stream.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now-last_progress >= progress_seconds:
                            report('downloading', offset+received, attempt=attempt)
                            last_progress = now
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != expected_response_bytes:
                    raise IOError(f'Incomplete response: received {received}/{expected_response_bytes} bytes')
            actual_size = partial.stat().st_size
            report('verifying_download', actual_size, attempt=attempt)
            if actual_size != size:
                raise IOError(f'Incomplete checkpoint: {actual_size}/{size} bytes')
            if sha256_file(partial) != sha256:
                partial.unlink()
                raise ValueError('Checkpoint SHA256 mismatch; discarded untrusted complete partial')
            os.replace(partial, destination)
            report('completed', size, attempt=attempt)
            return receipt('downloaded')
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last_error = exc
            present = partial.stat().st_size if partial.exists() else 0
            report('retry' if attempt < attempts else 'failed', present,
                   attempt=attempt, error=f'{type(exc).__name__}: {exc}')
            if attempt < attempts:
                sleep(min(2 ** (attempt-1), 5))
    raise RuntimeError(f'Teacher checkpoint preparation failed after {attempts} attempts: {last_error}') from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True,
                        help='Explicit output model.safetensors path; prefer cluster /cache, never an implicit PC cache.')
    parser.add_argument('--inspect-only', action='store_true', help='Print pinned provenance without filesystem changes or downloads.')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--progress-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.inspect_only:
        print(json.dumps(dict(source=URL, revision=REVISION, filename=FILENAME, size=SIZE,
                              sha256=SHA256, destination=str(Path(args.destination).expanduser().resolve()),
                              download_started=False), indent=2))
        return
    prepare(args.destination, attempts=args.attempts, timeout=args.timeout,
            progress_seconds=args.progress_seconds)


if __name__ == '__main__':
    main()
