import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils.native_audit_io import aligned_indices, recover_case, recover_json, seal, safe_relative, validate_wan_checkpoint
from utils.native_wan_runtime import attention, compact_storage, rope_params, rope_apply, native_block_forward


class NativeAuditTests(unittest.TestCase):
    def test_one_second_not_full_video_time_stretch(self):
        self.assertEqual(aligned_indices(81), list(range(0,17,2)))
        with self.assertRaises(ValueError):
            aligned_indices(9)

    def test_varlen_attention_ignores_padded_keys_and_returns_padded_queries(self):
        torch.manual_seed(1)
        q=torch.randn(2,7,2,8); k=torch.randn(2,9,2,8); v=torch.randn_like(k)
        actual=attention(q,k,v,q_lens=[7,4],k_lens=[6,3],softmax_scale=.4,q_scale=.5)
        for i,(nq,nk) in enumerate(((7,6),(4,3))):
            scores=torch.einsum('qhd,khd->hqk',q[i,:nq]*.5,k[i,:nk])*.4
            expected=torch.einsum('hqk,khd->qhd',scores.softmax(-1),v[i,:nk])
            torch.testing.assert_close(actual[i,:nq],expected)
        self.assertEqual(float(actual[1,4:].abs().sum()),0.)
        k[1,3:]=1000;v[1,3:]=-1000
        torch.testing.assert_close(actual,attention(q,k,v,q_lens=[7,4],k_lens=[6,3],softmax_scale=.4,q_scale=.5))

    def test_causal_bottom_right(self):
        q=torch.ones(1,2,1,4); k=torch.ones(1,4,1,4)
        v=torch.arange(4.).view(1,4,1,1).expand(1,4,1,4)
        result=attention(q,k,v,causal=True)
        torch.testing.assert_close(result[0,:,0,0],torch.tensor([1.,1.5]))

    def test_rope_matches_complex_native_reference(self):
        torch.manual_seed(2)
        x=torch.randn(1,14,2,16)
        grid=torch.tensor([[2,2,3]])
        freqs=rope_params(16,16)
        parts=torch.view_as_complex(freqs.double().contiguous()).split([4,2,2],1)
        phase=torch.cat((parts[0][:2,None,None,:].expand(2,2,3,-1),
                         parts[1][None,:2,None,:].expand(2,2,3,-1),
                         parts[2][None,None,:3,:].expand(2,2,3,-1)),3).reshape(12,1,8)
        values=torch.view_as_complex(x[0,:12].double().reshape(12,2,8,2))
        expected=torch.cat((torch.view_as_real(values*phase).flatten(2).float(),x[0,12:]),0)[None]
        torch.testing.assert_close(rope_apply(x,grid,freqs),expected,atol=1e-6,rtol=1e-5)

    def test_storage_preserves_precision_islands_and_keys(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.LayerNorm(8))
                self.head=torch.nn.Linear(8,8)
                self.time_embedding=torch.nn.Linear(8,8)
                self.time_projection=torch.nn.Linear(8,8)
        model=Toy(); keys=set(model.state_dict())
        compact_storage(model)
        self.assertEqual(model.blocks[0].weight.dtype,torch.bfloat16)
        for m in (model.blocks[1],model.head,model.time_embedding,model.time_projection):
            self.assertEqual(m.weight.dtype,torch.float32)
        self.assertEqual(keys,set(model.state_dict()))

    def test_native_modulation_keeps_fp32(self):
        class Block:
            modulation=torch.ones(1,6,4)*.1234567
            norm1=norm2=norm3=staticmethod(lambda x:x)
            self_attn=staticmethod(lambda x,*a:x*.31)
            cross_attn=staticmethod(lambda x,*a:x*.11)
            ffn=staticmethod(lambda x:x*.23)
        x=torch.ones(1,3,4,dtype=torch.bfloat16)*.75
        e=torch.ones(1,6,4)*.9876543
        result=native_block_forward(Block(),x,e,None,None,None,None,None)
        factors=(Block.modulation+e).chunk(6,1)
        z=x.float()+((x.float()*(1+factors[1])+factors[0])*.31)*factors[2]
        z=z+z*.11
        expected=z+((z*(1+factors[4])+factors[3])*.23)*factors[5]
        torch.testing.assert_close(result,expected)
        self.assertEqual(result.dtype,torch.float32)

    def test_resume_uses_second_copy_for_corrupt_first(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); src=base/'src'; a=base/'a'; b=base/'b'; out=base/'out'
            for p in (src,a,b,out):p.mkdir()
            (src/'video.bin').write_bytes(b'good artifact')
            receipt=seal(src,'done.json','abc',['video.bin'],dict(seed=101))
            for p in (a,b):(p/'done.json').write_text(json.dumps(receipt))
            (a/'video.bin').write_bytes(b'corrupt');(b/'video.bin').write_bytes(b'good artifact')
            recovered=recover_case(out,'done.json','abc',[str(a),str(b)])
            self.assertEqual(recovered['metadata']['seed'],101)
            self.assertEqual((out/'video.bin').read_bytes(),b'good artifact')
            with self.assertRaises(ValueError):recover_case(out,'done.json','different',[str(a)])

    def test_missing_receipt_vs_access_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(recover_json('missing.json',folder,[]))
            with patch('utils.native_audit_io.exists',side_effect=PermissionError('blocked')):
                with self.assertRaises(RuntimeError):recover_json('missing.json',folder,[])

    def test_paths_and_t2v_rejected(self):
        for name in ('../a','/a','a\\b','obs://other'):
            with self.assertRaises(ValueError):safe_relative(name)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'config.json').write_text(json.dumps(dict(model_type='t2v')))
            with self.assertRaises(ValueError):validate_wan_checkpoint(folder)

    def test_sharded_checkpoint_requires_all_parts_and_tokenizers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'config.json').write_text(json.dumps(dict(model_type='i2v',dim=5120,
                num_layers=40,num_heads=40,in_dim=36,out_dim=16)))
            (root/'diffusion_pytorch_model.safetensors.index.json').write_text(
                json.dumps(dict(weight_map={'a':'part1.safetensors','b':'part2.safetensors'})))
            for name in ('part1.safetensors','Wan2.1_VAE.pth',
                         'models_t5_umt5-xxl-enc-bf16.pth',
                         'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                         'google/umt5-xxl/spiece.model','xlm-roberta-large/tokenizer.json'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            with self.assertRaises(FileNotFoundError):validate_wan_checkpoint(folder)
            (root/'part2.safetensors').write_bytes(b'fixture')
            self.assertIn('part2.safetensors',validate_wan_checkpoint(folder)['files'])
            (root/'google/umt5-xxl/spiece.model').unlink()
            with self.assertRaises(FileNotFoundError):validate_wan_checkpoint(folder)

    def test_mp4_keeps_518_dimensions(self):
        import imageio.v2 as imageio
        from audit_native_i2v import write_video
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'preview.mp4'
            write_video(path,torch.zeros(2,518,518,3,dtype=torch.uint8),8)
            reader=imageio.get_reader(str(path))
            try:self.assertEqual(reader.get_data(0).shape,(518,518,3))
            finally:reader.close()
            self.assertFalse(list(Path(folder).glob('*.pending.mp4')))


if __name__=='__main__':unittest.main()
