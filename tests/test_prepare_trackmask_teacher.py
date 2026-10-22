"""Use a local HTTP fixture; never download public VGGT weights in tests."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from scripts.prepare_trackmask_teacher import prepare, URL, SIZE, SHA256


REPO = Path(__file__).resolve().parents[1]
DATA = bytes(range(256))*37
DIGEST = hashlib.sha256(DATA).hexdigest()


@contextmanager
def serving(*, ignore_range=False, corrupt=False, truncate=False, bad_range=False):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get('Range')
            requests.append(requested)
            start = int(requested[6:-1]) if requested and not ignore_range else 0
            payload = (b'x'*len(DATA) if corrupt else DATA)[start:]
            code = 206 if requested and not ignore_range else 200
            self.send_response(code)
            self.send_header('Content-Length', str(len(payload)))
            if code == 206:
                begin = start+1 if bad_range else start
                self.send_header('Content-Range', f'bytes {begin}-{len(DATA)-1}/{len(DATA)}')
            self.end_headers()
            self.wfile.write(payload[:len(payload)//2] if truncate else payload)
            self.wfile.flush()
            self.close_connection = True
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/teacher', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class PrepareTrackmaskTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='teacher download ',
                                                    dir=REPO.parent if os.name == 'nt' else None)
        self.root = Path(self.temporary.name)
        self.destination = self.root/'weights/model.safetensors'
        self.events = []

    def tearDown(self):
        self.temporary.cleanup()

    def run_prepare(self, url, **kwargs):
        return prepare(self.destination, url=url, size=len(DATA), sha256=DIGEST,
                       chunk_size=193, progress_seconds=.001, emit=self.events.append,
                       sleep=lambda seconds: None, **kwargs)

    def partial(self, payload):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        path = self.destination.with_name(self.destination.name+'.partial')
        path.write_bytes(payload)
        return path

    def test_full_download_is_verified_then_atomically_promoted(self):
        with serving() as (url, requests):
            result = self.run_prepare(url)
        self.assertEqual(requests, [None])
        self.assertEqual(result['status'], 'downloaded')
        self.assertEqual(self.destination.read_bytes(), DATA)
        self.assertFalse(self.destination.with_name(self.destination.name+'.partial').exists())
        self.assertTrue(all(e['DI_throughput'] == 0 for e in self.events))
        receipt = json.loads(self.destination.with_name(self.destination.name+'.receipt.json').read_text())
        self.assertEqual(receipt['sha256'], DIGEST)

    def test_existing_valid_file_uses_no_network(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(DATA)
        result = self.run_prepare('http://127.0.0.1:1/never-open')
        self.assertEqual(result['status'], 'already_verified')

    def test_honored_range_resumes_partial(self):
        self.partial(DATA[:319])
        with serving() as (url, requests):
            self.run_prepare(url)
        self.assertEqual(requests, ['bytes=319-'])
        self.assertEqual(self.destination.read_bytes(), DATA)

    def test_ignored_range_restarts_instead_of_appending(self):
        self.partial(DATA[:319])
        with serving(ignore_range=True) as (url, requests):
            self.run_prepare(url)
        self.assertEqual(requests, ['bytes=319-'])
        self.assertEqual(self.destination.read_bytes(), DATA)
        self.assertIn('range_ignored_restart', [e['phase'] for e in self.events])

    def test_bad_content_range_never_appends_or_promotes(self):
        partial = self.partial(DATA[:319])
        with serving(bad_range=True) as (url, requests):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected partial range'):
                self.run_prepare(url, attempts=2)
        self.assertEqual(requests, ['bytes=319-']*2)
        self.assertEqual(partial.read_bytes(), DATA[:319])
        self.assertFalse(self.destination.exists())

    def test_truncated_response_keeps_partial_but_never_final(self):
        with serving(truncate=True) as (url, requests):
            with self.assertRaises(RuntimeError):
                self.run_prepare(url, attempts=2)
        self.assertEqual(len(requests), 2)
        self.assertIsNone(requests[0])
        self.assertIsNotNone(requests[1])
        self.assertFalse(self.destination.exists())
        partial = self.destination.with_name(self.destination.name+'.partial')
        self.assertGreater(partial.stat().st_size, 0)
        self.assertLess(partial.stat().st_size, len(DATA))

    def test_corrupt_full_download_never_replaces_existing_file(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'previous-invalid-file')
        with serving(corrupt=True) as (url, requests):
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                self.run_prepare(url, attempts=2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.destination.read_bytes(), b'previous-invalid-file')
        self.assertFalse(self.destination.with_name(self.destination.name+'.partial').exists())

    def test_complete_verified_partial_promotes_without_network(self):
        self.partial(DATA)
        self.run_prepare('http://127.0.0.1:1/never-open')
        self.assertEqual(self.destination.read_bytes(), DATA)

    def test_inspect_only_prints_pin_without_creating_destination(self):
        result = subprocess.run([sys.executable, str(REPO/'scripts/prepare_trackmask_teacher.py'),
                                 '--destination', str(self.destination), '--inspect-only'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual((info['source'], info['size'], info['sha256']), (URL, SIZE, SHA256))
        self.assertFalse(info['download_started'])
        self.assertFalse(self.destination.parent.exists())

    def test_destination_is_required(self):
        result = subprocess.run([sys.executable, str(REPO/'scripts/prepare_trackmask_teacher.py'),
                                 '--inspect-only'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--destination', result.stderr)


if __name__ == '__main__':
    unittest.main()
