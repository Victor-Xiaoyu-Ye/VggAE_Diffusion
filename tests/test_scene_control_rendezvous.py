"""Real TCPStore processes: missing, delayed and misconfigured startup peers."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from scripts.coordinated_scene_controls import connect_and_join, rendezvous, run


def worker(index, directory, mode):
    torch.set_num_threads(1)
    root = Path(directory)
    # A second node accidentally configured as rank1 must not count as rank2.
    rank = 1 if mode == 'duplicate' and index == 2 else index
    ready = root/'port.json'
    os.environ.update(NODE_RANK=str(rank), NNODES='3', MASTER_ADDR='127.0.0.1', MASTER_PORT='0')
    status_file = root/f'status{index}.json'
    text = io.StringIO()
    real_join = rendezvous

    def capture_port(store, *args):
        if index == 0:
            temp = root/'port.tmp'
            temp.write_text(json.dumps(store.port))
            temp.replace(ready)
        return real_join(store, *args)

    if index:
        deadline = time.monotonic()+25
        while not ready.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('owner never started listening')
            time.sleep(.02)
        os.environ['MASTER_PORT'] = str(json.loads(ready.read_text()))
        if mode == 'delayed' and index == 2:
            # Wait until the owner has reported the actual missing node.
            while not (root/'status0.json').is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError('owner did not report startup status')
                time.sleep(.02)
            time.sleep(.5)
    identity = 'pilot:smoke' if mode == 'mismatch' and index == 2 else 'pilot:all'
    with redirect_stdout(text), patch('scripts.coordinated_scene_controls.rendezvous', side_effect=capture_port):
        try:
            store, peers = connect_and_join(rank, 3, 2 if mode == 'missing' else 10,
                                            .05, 'index', identity, status_file)
            # This marker is written by a real child and must NEVER exist when
            # even one node failed to register, regardless of local readiness.
            command = [sys.executable, '-c',
                       'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ran")',
                       str(root/f'child{index}')]
            code = run(command, store, rank, 3, 10, .05, .3, phase='index',
                       status_file=status_file, joined_nodes=peers, run_id=identity)
            outcome = dict(code=code, error=None)
        except Exception as exc:
            outcome = dict(code=1, error=str(exc))
    (root/f'result{index}.json').write_text(json.dumps(outcome))
    (root/f'log{index}.txt').write_text(text.getvalue())


class RendezvousProcessTests(unittest.TestCase):
    def execute(self, mode):
        count = 2 if mode == 'missing' else 3
        with tempfile.TemporaryDirectory(prefix='coordinator join ', dir=Path(__file__).resolve().parents[2]) as temp:
            root = Path(temp)
            context = mp.spawn(worker, args=(temp, mode), nprocs=count, join=False)
            deadline = time.monotonic()+45
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail('startup coordination hung')
                return [(json.loads((root/f'result{i}.json').read_text()),
                         json.loads((root/f'status{i}.json').read_text()),
                         (root/f'child{i}').exists(), (root/f'log{i}.txt').read_text())
                        for i in range(count)]
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)

    def test_two_of_three_reports_missing_rank_without_starting_children(self):
        for result, status, ran, log in self.execute('missing'):
            self.assertEqual(result['code'], 1)
            self.assertIn('missing=[2]', result['error'])
            self.assertEqual(status['state'], 'join_failed')
            self.assertIsNone(status['local_exit'])
            self.assertEqual(status['missing_nodes'], [2])
            self.assertFalse(ran)
            self.assertIn('child not started', log)

    def test_delayed_third_node_unblocks_all_three(self):
        rows = self.execute('delayed')
        for result, status, ran, _ in rows:
            self.assertEqual(result, dict(code=0, error=None))
            self.assertEqual(status['state'], 'completed')
            self.assertEqual(set(status['joined_nodes']), {'0', '1', '2'})
            self.assertTrue(ran)
        self.assertIn('state=joining', rows[0][3])
        self.assertIn('2', rows[0][3])

    def test_smoke_and_all_cannot_join_the_same_phase(self):
        for result, status, ran, _ in self.execute('mismatch'):
            self.assertEqual(result['code'], 1)
            self.assertIn('identity mismatch', result['error'])
            self.assertEqual(status['phase_exit'], 1)
            self.assertFalse(ran)

    def test_duplicate_rank_cannot_replace_the_missing_rank(self):
        rows = self.execute('duplicate')
        self.assertIn('Duplicate NODE_RANK=1', rows[0][0]['error'])
        for result, _, ran, _ in rows:
            self.assertEqual(result['code'], 1)
            self.assertFalse(ran)


if __name__ == '__main__':
    unittest.main()
