"""Actual small codec/DiT optimization, exact resume and immutable-bank IO."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_scene_matched as trainer
from tests import test_scene_prefix_train as prefix_helpers
from utils.scene_run import ArtifactStore


def fixture(root):
    trainer.seed_all(151)
    data = ArtifactStore(root/'data', [str(root/'data-mirror')])
    bank = dict(identity='matched-fixture-bank', train=[], dev=[], train_eval=[],
                feature_dims=dict(dino=8,stream=16),views=9,size=8,grid=2)
    for split, count in [('train',3),('dev',2)]:
        for i in range(count):
            name=f'data/{split}/{i}.pt'; ident=f'{split}-{i}'
            sample=dict(id=ident,dataset='fixture',frame_ids=list(range(9)),
                        dino=torch.randn(9,4,8),stream=torch.randn(9,4,16),
                        raw=torch.randint(0,256,(9,3,8,8),dtype=torch.uint8))
            data.save(name,sample,bank['identity'])
            bank[split].append(dict(file=name,id=ident))
    bank['train_eval']=bank['train'][:2]
    data.write_json('data/bank.json',bank)
    cfg=dict(seed=42,arm='dino',views=9,history=4,size=8,grid=2,channels=4,
        decoder=dict(decoder_width=8,decoder_temporal_blocks=2,decoder_resblocks=0,use_checkpoint=True),
        codec=dict(steps=3,lr=.001,min_lr=.0001,accum=2,warmup_steps=1,
                   lpips_weight=0.,feature_weight=.1,temporal_weight=.1),
        flow_training=dict(steps=3,lr=.001,min_lr=.0001,accum=2,warmup_steps=1),
        flow=dict(width=24,depth=1,head_width=48,head_depth=1,heads=4,text_dim=8,checkpoint_blocks=True),
        log_every=1,save_every_steps=2,eval_every_steps=2,monitor_cases=1,
        sample_steps=2,eval_seeds=[101,202])
    return data,bank,cfg


fake_preview=prefix_helpers.fake_preview
record_metrics=prefix_helpers.record_metrics


def execute(data,bank,cfg,out,stage,stop=0,preview_fn=fake_preview,rank=0,world=1):
    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer,'preview',preview_fn))
        stack.enter_context(patch.object(trainer,'metrics',record_metrics))
        stack.enter_context(redirect_stdout(io.StringIO()))
        args=SimpleNamespace(stage=stage,arm=cfg['arm'],stop_after=stop)
        trainer.run_training(args,cfg,bank,data,out,'matched-run',torch.device('cpu'),rank,world)
    return out.load(f'{stage}/checkpoint_latest.pt','matched-run',True)


class MatchedTrainingTests(unittest.TestCase):
    assert_nested_equal=prefix_helpers.PrefixTrainingTests.assert_nested_equal

    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_codec_and_flow_exact_resume_and_train_only_stats(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data,bank,cfg=fixture(root)
            direct=ArtifactStore(root/'direct',[])
            resumed=ArtifactStore(root/'resume',[str(root/'primary'),str(root/'mirror')])
            for stage in ('codec','flow'):
                a=execute(data,bank,cfg,direct,stage)
                first=execute(data,bank,cfg,resumed,stage,2)
                self.assertEqual(first['step'],2)
                b=execute(data,bank,cfg,resumed,stage)
                for key in ('model','optimizer','ema','rng','cursors'):
                    self.assert_nested_equal(a[key],b[key])
                self.assertEqual(b['ema_metadata']['num_updates'],3)
                self.assertEqual(resumed.json(f'{stage}/complete.json')['evaluation_status'],'completed')
                if stage=='codec':
                    for out in (direct,resumed):
                        codec=trainer.load_codec(out,cfg,8,'matched-run','cpu')
                        with patch.object(trainer,'metrics',record_metrics):
                            trainer.cache_latents(codec,data,bank,out,cfg,'matched-run',torch.device('cpu'),0,1)
                        manifest=out.json('cache/complete.json',True)
                        train=torch.cat([out.load(e['file'],manifest['identity'],True)['target'].flatten(0,1)
                                         for e in manifest['train']]).double()
                        torch.testing.assert_close(torch.tensor(manifest['mean']),train.mean(0).float())
                        self.assertEqual(len(manifest['train_eval']),2)
                        self.assertEqual({e['id'] for e in manifest['train_eval']},{e['id'] for e in bank['train_eval']})
            summary=resumed.json('flow/eval/step0000003/summary.json',True)
            self.assertEqual(len(summary['rows']),16) # 4 cases x 2 weights x 2 seeds
            self.assertEqual({r['split'] for r in summary['rows']},{'train_eval','dev'})
            self.assertTrue(all('normalized_latent_mse' in r and 'future_geo_mse' not in r for r in summary['rows']))
            self.assertTrue(all(r['DI_throughput']>0 for r in summary['rows']))

    def test_media_failure_repair_does_not_reserialize_selected_codec(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data,bank,cfg=fixture(root); out=ArtifactStore(root/'out',[])
            a=execute(data,bank,cfg,out,'codec',preview_fn=lambda *_: [])
            self.assertEqual(out.json('codec/complete.json')['evaluation_status'],'partial_media')
            before=out.json('codec/selected.pt.json')
            b=execute(data,bank,cfg,out,'codec')
            self.assertEqual(before,out.json('codec/selected.pt.json'))
            self.assert_nested_equal(a['model'],b['model'])
            self.assertEqual(out.json('codec/complete.json')['evaluation_status'],'completed')

    def test_immutable_cache_repair_and_local_weight_retention(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); source=ArtifactStore(root/'primary',[str(root/'mirror')])
            source.save('data/one.pt',dict(value=torch.tensor([7.])), 'id')
            data=trainer.ImmutableBank(root/'local',[str(source.local),str(root/'mirror')])
            data.load('data/one.pt','id',True)
            (data.local/'data/one.pt').write_bytes(b'corrupt')
            (source.local/'data/one.pt').write_bytes(b'corrupt')
            self.assertEqual(data.load('data/one.pt','id',True)['value'].item(),7.)
            out=ArtifactStore(root/'out',[str(root/'durable')])
            for i in range(5): out.save(f'flow/weights_step{i:07d}.pt',dict(step=i),'id')
            out.save('flow/checkpoint_latest.pt',dict(step=4),'id')
            trainer.prune_local_weights(out,'flow',2)
            self.assertEqual(len(list((out.local/'flow').glob('weights_step*.pt'))),2)
            self.assertTrue((out.local/'flow/checkpoint_latest.pt').exists())
            self.assertEqual(len(list((root/'durable/flow').glob('weights_step*.pt'))),5)
            self.assertEqual(out.load('flow/weights_step0000000.pt','id',True)['step'],0)


if __name__=='__main__': unittest.main()
