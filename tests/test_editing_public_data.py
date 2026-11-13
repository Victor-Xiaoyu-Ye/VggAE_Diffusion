"""Real encoded public-format videos: time/FOV/mask and source-only selection."""
import json
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np
import torch

from utils.editing_public_data import (audit_index, public_source, read_target,
    freeze_public, exposed_benchmark_ids, color_metrics)
from utils.editing_paper_data import source_fingerprint

ROOT=Path(__file__).resolve().parents[1]
GEOMETRY=dict(height=16,width=32,max_side=32,frames=17,fps=8)


def write_movie(path,frames,fps=8):
    path.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimwrite(str(path),frames,fps=fps,codec='libx264',macro_block_size=1,
                    pixelformat='yuv444p',ffmpeg_params=['-crf','0'])


def make_benchmark(root,count=1):
    root=Path(root); (root/'indices').mkdir(parents=True)
    rows=[]; yy,xx=np.mgrid[:16,:32]
    for i in range(count):
        frames=np.stack([30+(xx*5+yy*3+t*2+i*19)%170 for t in range(17)]).astype('uint8')
        gray=np.repeat(frames[...,None],3,axis=-1)
        color=gray.copy(); color[...,0]=np.minimum(color[...,0].astype(int)+25,255)
        row=dict(source='real' if i%2==0 else 'synthetic',sample_id=f'{40+i:06d}',task='V2V',subtask='gray',
            src_video=f'data/gray{i}.mp4',src_mask=f'data/mask{i}.mp4',raw_video=f'data/raw{i}.mp4',
            en_prompt='A red surface moving to the right.',src_ref_images=[])
        for name,value in ((row['src_video'],gray),(row['raw_video'],color),(row['src_mask'],np.full_like(gray,255))):
            write_movie(root/name,value)
        rows.append(row)
    for domain in ('real','synthetic','composition'):
        (root/'indices'/f'{domain}.txt').write_text('\n'.join(json.dumps(r) for r in rows if r['source']==domain)+'\n',encoding='utf8')
    return rows


class PublicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='public data ',dir=ROOT.parent)
        self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        self.raw=make_benchmark(self.root,2)

    def test_target_only_evaluation_and_source_geometry(self):
        rows,labels,audit=audit_index(self.root)
        self.assertEqual(audit['eligible_domains'],dict(real=1,synthetic=1))
        self.assertTrue(all('target' not in r and 'raw_video' not in r for r in rows))
        path=self.root/labels[rows[0]['id']]['target']; target=path.read_bytes(); path.unlink()
        item=public_source(self.root,rows[0],GEOMETRY)
        self.assertEqual(item['source'].shape,(17,3,16,32))
        self.assertEqual(item['timing']['indices'],list(range(17)))
        self.assertTrue(torch.equal(item['native_mask'],torch.full((17,1,16,32),255,dtype=torch.uint8)))
        with self.assertRaises(FileNotFoundError): read_target(self.root,labels[rows[0]['id']],item,GEOMETRY)
        path.write_bytes(target)
        color,meta=read_target(self.root,labels[rows[0]['id']],item,GEOMETRY)
        self.assertEqual(color.shape,item['source'].shape)
        self.assertEqual(meta['indices'],item['timing']['indices'])
        values=color_metrics(item['source'],color,color)
        self.assertEqual(values['target_lab_ab_l1'],0.)
        self.assertGreater(color_metrics(item['source'],color,item['source'])['target_lab_ab_l1'],0.)

    def test_invalid_official_mask_and_short_source_are_not_stretched(self):
        rows,_,_=audit_index(self.root)
        mask=np.zeros((17,16,32,3),dtype='uint8')
        write_movie(self.root/rows[0]['mask'],mask)
        with self.assertRaisesRegex(ValueError,'all-white'): public_source(self.root,rows[0],GEOMETRY)
        write_movie(self.root/rows[1]['source'],np.full((5,16,32,3),120,dtype='uint8'))
        with self.assertRaisesRegex(ValueError,'too short'): public_source(self.root,rows[1],GEOMETRY)

    def test_old_ids_cross_task_asset_and_visual_duplicates_excluded(self):
        rows,_,audit=audit_index(self.root); reports=[]
        for row in rows:
            x=public_source(self.root,row,GEOMETRY)
            reports.append(dict(id=row['id'],status='completed',group=x['group'],file='source.pt',
                fingerprint=source_fingerprint(x['source'],x['spatial']['valid_rect'])))
        old=exposed_benchmark_ids(dict(records=[dict(id='benchmark-real-000055',split='benchmark_dev')]))
        self.assertEqual(old,{'vace-real-000055'})
        assets=dict(audit['asset_groups'],**{'vace-real-000055':audit['asset_groups'][rows[0]['id']]})
        frozen=freeze_public(rows,reports,[],old,20,asset_groups=assets)
        self.assertNotIn(rows[0]['id'],[r['id'] for r in frozen['records']])
        self.assertIn('same_original_asset',frozen['excluded'][0]['reason'] if frozen['excluded'] else '')
        exposed=[dict(reports[1],id='different-file-id',group='other-byte-hash')]
        frozen=freeze_public(rows,reports,exposed,set(),20)
        self.assertNotIn(rows[1]['id'],[r['id'] for r in frozen['records']])
        again=freeze_public(list(reversed(rows)),list(reversed(reports)),exposed,set(),20)
        self.assertEqual(frozen,again)

    def test_original_target_metadata_cannot_change_fov_or_time(self):
        rows,labels,_=audit_index(self.root); row=rows[0]
        item=public_source(self.root,row,GEOMETRY)
        write_movie(self.root/labels[row['id']]['target'],np.full((17,32,16,3),120,dtype='uint8'))
        with self.assertRaisesRegex(ValueError,'FOV'): read_target(self.root,labels[row['id']],item,GEOMETRY)


if __name__=='__main__': unittest.main()
