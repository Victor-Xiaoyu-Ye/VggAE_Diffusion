import unittest
import tempfile
import json
from pathlib import Path
import torch
from utils.window_flow import WindowFlow
from utils.window_trajectory import capture, run_audit, NODES


class TrajectoryTests(unittest.TestCase):
    def test_capture_matches_production_and_probe_does_not_change_path(self):
        torch.manual_seed(1)
        y, noise, c = torch.randn(1,2,4,6),torch.randn(1,2,4,6),torch.randn(1,1,4,6)
        def model(x,u,c,*args): return .2*x+.1*c+.3*u[:,None,None,None]
        for dtype in (torch.float32,torch.bfloat16):
            flow=WindowFlow()
            expected=flow.sample(model,c,noise,64,dtype)
            snapshots,final=capture(model,flow,c,y,noise,dtype)
            torch.testing.assert_close(final,expected,atol=0,rtol=0)
            self.assertEqual(tuple(snapshots),NODES)
            torch.testing.assert_close(snapshots[0]['x0'],snapshots[0]['gt_probe'])
            torch.testing.assert_close(snapshots[0]['state'],noise)

    def test_metric_matrix_and_latents_before_preview(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__();self.p=torch.nn.Parameter(torch.tensor(.1))
            def forward(self,x,u,c,*args): return self.p*x
        model=Model()
        sample=dict(video_id='a',cond=torch.zeros(1,4,6),target=torch.ones(2,4,6))
        stats=dict(cond=dict(mean=torch.zeros(1,6),std=torch.ones(1,6)),
                   target=dict(mean=torch.zeros(2,6),std=torch.ones(2,6)))
        def decode(c,y):return torch.cat((c,y),1).reshape(1,3,2,2,6)[...,:3].sigmoid()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            run_audit(model=model,flow=WindowFlow(),online_state=model.state_dict(),
                data={'memorization':[sample],'heldout':[dict(sample,video_id='b')]},memory_count=64,
                model_args=dict(future=2,grid=2,channels=6),stats=stats,decode=decode,
                dtype=torch.float32,device=torch.device('cpu'),rank=0,world=1,ddp=False,
                out=out,seeds=[101],previews=0,status=lambda *a,**k:None)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['rows'],112)
            self.assertTrue(summary['complete'])
            self.assertEqual(len(list((out/'latents').glob('*.pt'))),4)


if __name__=='__main__':unittest.main()
