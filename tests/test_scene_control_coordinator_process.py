"""Actual three-process TCPStore lifetime, including owner exit and spawn failure.

The store owner is destroyed with its process, unlike a long-lived thread test
fixture. No NPU, HCCL, dataset or external network is involved.
"""
from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scripts.coordinated_scene_controls import run


class DelayedAck:
    def __init__(self,store): self.store=store
    def __getattr__(self,name): return getattr(self.store,name)
    def set(self,key,value):
        if key=='ack2': time.sleep(.35)
        return self.store.set(key,value)


def worker(rank,directory,mode):
    torch.set_num_threads(1)
    root=Path(directory)
    ready=root/'port.json'
    if rank==0:
        store=dist.TCPStore('127.0.0.1',0,3,True,timedelta(seconds=10),
                            wait_for_workers=False,use_libuv=False)
        temp=root/'port.tmp'; temp.write_text(json.dumps(store.port));temp.replace(ready)
    else:
        deadline=time.monotonic()+30
        while not ready.is_file():
            if time.monotonic()>deadline: raise TimeoutError('owner did not start')
            time.sleep(.02)
        port=json.loads(ready.read_text())
        store=dist.TCPStore('127.0.0.1',port,3,False,timedelta(seconds=10),use_libuv=False)
    if mode=='disconnect' and rank==0:
        # Both children really finished successfully, but no global phase
        # result exists: simulate loss of the server owner at that point.
        store.wait(['result1','result2'],timedelta(seconds=20))
        return
    if (mode=='leader_spawn_failure' and rank==0) or (mode=='follower_spawn_failure' and rank==1):
        command=['nonexistent-vggae-coordinator-test-executable']
    else:
        delay=.1*rank if mode!='follower_spawn_failure' else 1.
        command=[sys.executable,'-c',f'import time; time.sleep({delay})']
    text=io.StringIO()
    with redirect_stdout(text):
        try:
            code=run(command,DelayedAck(store),rank,3,20,.02,.3,
                phase=mode,status_file=root/f'status{rank}.json')
            outcome=dict(code=code,error=None)
        except Exception as exc:
            outcome=dict(code=1,error=type(exc).__name__)
    (root/f'result{rank}.json').write_text(json.dumps(outcome))
    (root/f'log{rank}.txt').write_text(text.getvalue())
    # Deliberately do not add a multiprocessing/test barrier here. The real
    # coordinator must keep node0 alive until the delayed client has finished.


class CoordinatorProcessTests(unittest.TestCase):
    def execute(self,mode):
        with tempfile.TemporaryDirectory(prefix='coordinator tcp ',dir=Path(__file__).resolve().parents[2]) as temp:
            context=mp.spawn(worker,args=(temp,mode),nprocs=3,join=False)
            deadline=time.monotonic()+50
            try:
                while not context.join(timeout=1):
                    if time.monotonic()>deadline: self.fail('coordinator processes hung')
                root=Path(temp)
                ranks=range(1,3) if mode=='disconnect' else range(3)
                return [(json.loads((root/f'result{rank}.json').read_text()),
                         json.loads((root/f'status{rank}.json').read_text())) for rank in ranks]
            finally:
                for process in context.processes:
                    if process.is_alive(): process.terminate()
                for process in context.processes: process.join(timeout=5)

    def test_success_owner_waits_for_delayed_follower(self):
        for result,status in self.execute('success'):
            self.assertEqual(result,dict(code=0,error=None))
            self.assertEqual(status['state'],'completed')
            self.assertTrue(status['all_results_observed'])

    def test_owner_spawn_failure_reaches_every_peer(self):
        for result,status in self.execute('leader_spawn_failure'):
            self.assertEqual(result,dict(code=1,error=None))
            self.assertEqual(status['results']['0'],1)
            self.assertTrue(status['all_results_observed'])
            if status['node']==0: self.assertIn('FileNotFoundError',status['local_error'])

    def test_follower_spawn_failure_reaches_every_peer(self):
        for result,status in self.execute('follower_spawn_failure'):
            self.assertEqual(result,dict(code=1,error=None))
            self.assertEqual(status['results']['1'],1)
            self.assertTrue(status['all_results_observed'])

    def test_real_disconnect_does_not_masquerade_as_success(self):
        for result,status in self.execute('disconnect'):
            self.assertEqual(result,dict(code=1,error='DistNetworkError'))
            self.assertEqual(status['local_exit'],0)
            self.assertEqual(status['phase_exit'],1)
            self.assertFalse(status['all_results_observed'])


if __name__=='__main__': unittest.main()
