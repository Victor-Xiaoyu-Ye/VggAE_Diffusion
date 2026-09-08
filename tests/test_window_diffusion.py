"""CPU contracts and exact interruption replay; synthetic results are not video quality evidence."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.r7_window_dit import R7WindowDiT
from utils.window_flow import WindowFlow
from utils.window_training import CaptionBank, validate_resume
from train_r7_window_diffusion import run, parse_args
from models.causal_dual_tokenizer import CausalDualTokenizerCore


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle(self):
        for future in (4, 8):
            y, noise = torch.randn(1, future, 4, 6), torch.randn(1, future, 4, 6)
            for prediction in ('x0', 'velocity'):
                flow = WindowFlow(prediction)
                def oracle(x, u, c, text, valid):
                    self.assertTrue(bool((u > 0).all()))
                    return y if prediction == 'x0' else noise-y
                for dtype in (torch.float32, torch.bfloat16):
                    for method in ('euler', 'heun'):
                        result = flow.sample(oracle, y[:, :1], noise, 7, dtype, method=method)
                        torch.testing.assert_close(result, y, atol=2e-6, rtol=2e-6)
            flow = WindowFlow()
            u = torch.tensor([.25])
            error = torch.ones_like(y)*.1
            self.assertAlmostEqual(float(flow.loss(y+error, y, noise, u)), .16, places=5)
        with self.assertRaises(RuntimeError):
            WindowFlow().sample(lambda *a: torch.full_like(noise,float('nan')), y[:,:1],noise,2)

    def test_real_tokenizer_independent_anchor(self):
        for factor in (1,2,4):
            codec=CausalDualTokenizerCore(geo_dim=4,tex_dim=4,geo_latent_dim=3,
                tex_latent_dim=3,temporal_factor=factor,temporal_depth=1).eval()
            # Nonzero residual weights exercise causality beyond identity init.
            with torch.no_grad():
                for name,p in codec.named_parameters():
                    if 'conv2.weight' in name: p.normal_(std=.01)
                geo,tex=torch.randn(1,9,2,2,4),torch.randn(1,9,2,2,4)
                full=codec.encode(geo,tex)
                single=codec.encode(geo[:,:1],tex[:,:1])
                torch.testing.assert_close(full[:,:1],single,atol=1e-6,rtol=1e-5)
                self.assertEqual(codec.decode(full)[0].shape,(1,9,2,2,4))
                self.assertEqual(codec.decode(single)[0].shape,(1,1,2,2,4))

    def test_full_window_condition_and_gradients(self):
        for factor in (1, 2):
            m = R7WindowDiT(channels=6, grid=2, future=8//factor, temporal_factor=factor,
                            width=24, depth=1, heads=3, text_dim=8)
            # Activate zero-initialized gates/head to test information paths.
            torch.nn.init.normal_(m.head.weight, std=.1)
            torch.nn.init.normal_(m.blocks[0].mod[-1].bias, std=.1)
            x = torch.randn(1,8//factor,4,6); c = torch.randn(1,1,4,6)
            t = torch.randn(1,3,8); valid = torch.tensor([[True, True, False]])
            u = torch.tensor([.5])
            a = m(x,u,c,t,valid)
            self.assertEqual(a.shape, x.shape)
            self.assertGreater(float((a-m(x,u,c+torch.randn_like(c),t,valid)).detach().abs().max()), 1e-5)
            self.assertGreater(float((a-m(x,u,c,t+1,valid)).detach().abs().max()), 1e-5)
            altered = t.clone(); altered[:,2] += 1000
            torch.testing.assert_close(a,m(x,u,c,altered,valid))
            a.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            m.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                mixed=m(x,u,c,t,valid)
            mixed.square().mean().backward()
            self.assertTrue(torch.isfinite(mixed).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_caption_missing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p/'_SUCCESS').write_text(json.dumps({'schema':'wan-umt5xxl-text-embeddings-v1','text_len':512}))
            (p/'index.json').write_text(json.dumps({'a':'part.pt'}))
            torch.save(torch.ones(2,4096),p/'empty_prompt.pt')
            torch.save({'a':torch.ones(3,4096)},p/'part.pt')
            bank = CaptionBank(td)
            self.assertEqual(bank.batch(['a'],'cpu')[0].shape,(1,3,4096))
            with self.assertRaises(ValueError): bank.batch(['missing'],'cpu')

    def test_resume_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            torch.manual_seed(19)
            for split in ('train','eval'):
                with tarfile.open(p/f'{split}.tar','w') as tar:
                    for i in range(5):
                        b = io.BytesIO()
                        torch.save(dict(cond=torch.randn(1,4,6), target=torch.randn(2,4,6),
                            video_id=f'{split}{i}', requested_video_id=f'{split}{i}'),b)
                        data=b.getvalue(); info=tarfile.TarInfo(f'{i}.pt'); info.size=len(data)
                        tar.addfile(info,io.BytesIO(data))
                (p/f'{split}.txt').write_text(str(p/f'{split}.tar'))
            stats = dict(representation={'config':dict(temporal_factor=1,seq_len=3,
                geo_latent_dim=3,tex_latent_dim=3,latent_grid=2)},
                cond={'mean':torch.zeros(1,6),'std':torch.ones(1,6)},
                target={'mean':torch.zeros(2,6),'std':torch.ones(2,6)})
            torch.save(stats,p/'stats.pt')
            base=['--manifest',str(p/'train.txt'),'--eval_manifest',str(p/'eval.txt'),
                '--stats',str(p/'stats.pt'),'--eval_stats',str(p/'stats.pt'),'--r7_ckpt','synthetic',
                '--no_text','--cpu_test','--dtype','fp32','--width','24','--depth','1','--heads','3',
                '--max_steps','4','--warmup_steps','0','--eval_every','4','--save_every','2',
                '--eval_clips','1','--preview_clips','1','--sample_steps','2','--sample_seeds','42',
                '--shuffle_buffer','3','--log_every','1']
            with contextlib.redirect_stdout(io.StringIO()):
                run(parse_args(base+['--output_dir',str(p/'full')]))
                run(parse_args(base+['--output_dir',str(p/'split'),'--stop_after_steps','2']))
                ckpt=str(p/'split/checkpoint_latest.pt')
                run(parse_args(base+['--output_dir',str(p/'split'),'--resume',ckpt]))
            full=torch.load(p/'full/checkpoint_final.pt',weights_only=False)
            resumed=torch.load(p/'split/checkpoint_final.pt',weights_only=False)
            for key in ('model','ema'):
                for name,value in full[key].items():
                    torch.testing.assert_close(value,resumed[key][name],atol=0,rtol=0)
            self.assertEqual(full['scheduler'],resumed['scheduler'])
            self.assertEqual(full['consumed_by_rank'],resumed['consumed_by_rank'])
            for key,value in full['optimizer']['state'].items():
                for name,tensor in value.items():
                    torch.testing.assert_close(tensor,resumed['optimizer']['state'][key][name],atol=0,rtol=0)
            self.assertEqual(json.loads((p/'split/run_status.json').read_text())['status'],'completed')
            self.assertTrue(list((p/'split/samples').rglob('*.pt')))
            with self.assertRaises(ValueError): validate_resume(resumed,{'wrong':'AE'})


if __name__ == '__main__':
    unittest.main()
