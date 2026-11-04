"""Two real Gloo workers prepare/solve independent cases and collect coverage."""
from datetime import timedelta
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.scene_run import ArtifactStore
from tests.test_edit_transport_probe import fixture,FakeRAE,FakeEncoder,FakeNative,REPO


def worker(rank,folder):
    import audit_edit_transport as audit
    root=Path(folder);torch.set_num_threads(1)
    os.environ['RANK']=str(rank);os.environ['LOCAL_RANK']=str(rank)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=2,
        timeout=timedelta(seconds=120))
    try:
        fixture_data=torch.load(str(root/'fixture.pt'),weights_only=False)
        cfg,args=fixture_data['cfg'],fixture_data['args']
        out=ArtifactStore(root/'local',[str(root/'primary'),str(root/'mirror')])
        vae=SimpleNamespace(model=FakeNative(),scale=[torch.tensor(0.),torch.tensor(1.)])
        with patch.object(audit,'load_teacher',side_effect=lambda *_:FakeRAE()),\
             patch.object(audit,'load_encoder',side_effect=lambda *_:FakeEncoder()),\
             patch('utils.vace_edit_runtime.load_vae',return_value=vae),contextlib.redirect_stdout(io.StringIO()):
            audit.prepare(args,cfg,ArtifactStore(root/'data',[]),out,torch.device('cpu'),rank,2)
            for phase in ('rae','vae'):
                args.phase=phase;audit.run_codec(args,cfg,out,torch.device('cpu'),rank,2)
            if rank==0:audit.summarize(out)
            dist.barrier()
    finally:dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(),'Gloo required')
class DistributedTransportTests(unittest.TestCase):
    def test_each_source_is_assigned_once_and_both_codecs_complete(self):
        with tempfile.TemporaryDirectory(prefix='transport gloo ',dir=REPO.parent) as temp:
            root=Path(temp);cfg,_,args=fixture(root,2)
            torch.save(dict(cfg=cfg,args=args),str(root/'fixture.pt'))
            mp.spawn(worker,args=(temp,),nprocs=2,join=True)
            report=ArtifactStore(root/'local',[]).json('summary.json',True)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['actual'],dict(rae=14,rgb=6,vae=2))
            self.assertEqual(report['source_completed'],2)


if __name__=='__main__':unittest.main()
