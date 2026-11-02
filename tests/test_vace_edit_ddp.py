"""Two CPU/Gloo ranks with actual tiny VACE, LoRA, RAE loss and process restart."""
from datetime import timedelta
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_vace_edit_pilot import cpu_native_import,tiny_native,config
from utils.vace_edit_runtime import EditingModel
from utils.scene_run import ArtifactStore


class Decoder(nn.Module):
    def __init__(self): super().__init__(); self.project=nn.Linear(256,3)
    def decode(self,z): return self.project(z)


def worker(rank,directory,phase):
    import train_vace_edit as trainer
    root=Path(directory)
    os.environ['RANK']=str(rank); torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(root/('rendezvous-'+phase)).as_uri(),
        rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        with cpu_native_import():
            torch.manual_seed(31)
            cfg=config(); cfg['accum']=1
            model=EditingModel(tiny_native().requires_grad_(False),cfg,True)
            decoder=Decoder().requires_grad_(False)
            fixture=torch.load(str(root/'fixture.pt'),weights_only=False)
            out=ArtifactStore(root/('direct' if phase=='direct' else 'resumed'),[])
            args=SimpleNamespace(arm='rae',stop_after=2 if phase=='preflight' else 0)
            with patch.object(trainer,'evaluate',return_value=[]):
                trainer.run_training(args,cfg,fixture['bank'],ArtifactStore(root/'data',[]),out,
                    'ddp-fixture',torch.device('cpu'),rank,2,model,fixture['teacher'],decoder)
            torch.save(model.deltas(),str(root/f'{phase}-rank{rank}.pt'))
            dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(),'Gloo required')
class EditingDDPTests(unittest.TestCase):
    def test_restart_keeps_ranks_and_updates_equal(self):
        with tempfile.TemporaryDirectory(prefix='editing ddp ',dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); store=ArtifactStore(root/'data',[])
            bank=dict(identity='native-fixture',records=[])
            teacher=dict(identity='teacher-fixture',records=[])
            torch.manual_seed(7)
            for i in range(4):
                ident=str(i); file=f'native/{i}.pt'
                store.save(file,dict(clean=torch.randn(16,2,4,4),context=torch.randn(96,2,4,4),
                    text=torch.randn(5,12),empty=torch.randn(1,12)),bank['identity'])
                bank['records'].append(dict(id=ident,file=file,split='train'))
                file=f'rae/{i}.pt'
                store.save(file,dict(z=torch.randn(2,18,18,256)),teacher['identity'])
                teacher['records'].append(dict(id=ident,file=file))
            torch.save(dict(bank=bank,teacher=teacher),str(root/'fixture.pt'))
            for phase in ('direct','preflight','resume'):
                mp.spawn(worker,args=(tmp,phase),nprocs=2,join=True)
            final=[torch.load(str(root/f'{phase}-rank{r}.pt'),weights_only=False)
                   for phase in ('direct','resume') for r in range(2)]
            for state in final[1:]:
                for key,value in final[0].items():
                    torch.testing.assert_close(value,state[key],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
