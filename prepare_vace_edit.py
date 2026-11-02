"""Offline staged Señorita -> native VACE cache -> optional frozen RAE targets."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import time

from utils.editing_data import digest, make_manifest, unpack_archive, load_pair


def stage_data(source,destination):
    from utils.moxing_io import is_remote_path,copy_file
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    if is_remote_path(source):
        import moxing
        names=moxing.file.list_directory(source,recursive=True)
        names=[n[len(source.rstrip('/'))+1:] if n.startswith(source.rstrip('/')+'/') else n for n in names]
    else:
        names=[p.relative_to(source).as_posix() for p in Path(source).rglob('*') if p.is_file()]
    # Only the agreed pilot packages. Never recursively materialize a 1.6TB dataset.
    allowed={str(i)+'.tar.gz' for i in (4,14,30,50,70,90,110,130,150,170,180,193)}
    selected=[n for n in names if Path(n).name=='updated_data_local_style_transfer.json' or
              (Path(n).name in allowed and ('local_style_transfer' in n or '/' not in n))]
    if not any(n.endswith('.json') for n in selected):
        raise FileNotFoundError('Pilot needs the official local_style_transfer metadata JSON')
    for name in sorted(selected):
        from utils.editing_data import safe_path
        safe_path(name)
        local=root/name; remote=source.rstrip('/')+'/'+name
        size=(moxing.file.stat(remote).length if is_remote_path(source) else Path(remote).stat().st_size)
        if not local.is_file() or local.stat().st_size!=size:
            print('[editing data] stage '+name,flush=True)
            copy_file(remote,str(local))
        if name.endswith('.tar.gz'):
            unpack_archive(local,root/'unpacked')
    print('[editing data] staged packages='+str(sum(n.endswith('.tar.gz') for n in selected)),flush=True)


def benchmark_records(root,cfg):
    rows=[]
    for index in sorted((Path(root)/'indices').glob('*.txt')):
        for line in index.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            if row.get('task') not in ('V2V','MV2V') or row.get('src_ref_images'):
                continue
            if not all(row.get(k) and (Path(root)/row[k]).is_file() for k in ('raw_video','src_video','src_mask')):
                continue
            rows.append(dict(id='benchmark-'+row['source']+'-'+row['sample_id'],
                source=row['src_video'],target=row['raw_video'],mask=row['src_mask'],
                instruction=row['en_prompt'],prompt_kind='official_output_description',
                split='benchmark_dev',task=row['task'],subtask=row['subtask'],group=row['sample_id']))
    # Balance tasks before deterministic selection; these few cases become dev, not test.
    rows.sort(key=lambda r:digest([cfg['seed'],r['id']]))
    rows=rows[:cfg['benchmark_cases']]
    from utils.editing_data import file_hash
    for row in rows:
        row['content_sha256']={k:file_hash(Path(root)/row[k]) for k in ('source','target','mask')}
    return rows


def run_index(a,cfg):
    from utils.vace_edit_runtime import validate_assets
    validate_assets(a.model)
    # Only metadata/inventory here; no SAM initialization, model downloads or target masks.
    sam=Path(a.model).parent/'sam2.1-hiera-large'
    print('[editing assets] '+json.dumps(dict(vace=str(a.model),benchmark=str(a.benchmark),
        sam2_native_checkpoint=(sam/'sam2.1_hiera_large.pt').is_file(),sam2_used=False)),flush=True)
    stage_data(a.source,a.data)
    manifest=make_manifest(a.data,cfg)
    manifest['benchmark']=benchmark_records(a.benchmark,cfg)
    manifest['identity']=digest(dict(manifest=manifest,geometry={k:cfg[k] for k in
        ('height','width','max_side','resize_policy','frames','fps','benchmark_fps','rae_frames')},negative_prompt=cfg['negative_prompt']))
    path=Path(a.data)/'manifest.json'
    if path.exists() and json.loads(path.read_text())['identity']!=manifest['identity']:
        print('[editing data] manifest changed; new bank identity (old caches cannot be silently reused)',flush=True)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('[editing manifest] '+json.dumps({k:manifest[k] for k in ('counts','available_pairs','available_groups','metadata_unavailable_pairs','identity')}),flush=True)
    if not manifest['counts']['train'] or not manifest['counts']['dev']:
        raise ValueError('No train/dev pairs after source-group split; check staged package coverage')


def cache(a,cfg):
    import torch
    import torch.distributed as dist
    from utils.editing_train import distributed,barrier,gather,progress,load_teacher,load_encoder
    from utils.scene_run import ArtifactStore
    from utils.vace_edit_runtime import validate_assets,load_text,load_vae,pack_context,runtime_preflight
    device,rank,world=distributed()
    store=ArtifactStore(a.output,a.root)
    store.write_json(f'runtime_preflight/rank{rank:03d}.json',runtime_preflight(device))
    manifest=json.loads((Path(a.data)/'manifest.json').read_text(encoding='utf8'))
    signatures=validate_assets(a.model) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    signatures=gather(signatures,world)
    active=[s for s in signatures if s is not None]
    if any(s!=active[0] for s in active): raise ValueError('Nodes have different VACE weights')
    identity=digest(dict(manifest=manifest['identity'],model=active[0],schema='native-vace-cache-v1'))
    if any(x!=identity for x in gather(identity,world)): raise ValueError('Nodes disagree on data/cache identity')
    tasks=[r for r in manifest['records'] if r['split']!='test']+manifest['benchmark']
    reports,bad=[],[]
    text=vae=None
    started=time.monotonic(); completed=0
    for row in tasks[rank::world]:
        name='native/'+row['id']+'.pt'
        old=store.load(name,identity)
        if old is not None:
            reports.append(dict(row,file=name,spatial=old['timing']['spatial'],
                original_fps=old['timing']['original_fps'],quality=old.get('quality'))); del old
            continue
        # Data failures skip a pair, but model setup/forward errors must remain visible.
        try:
            root=a.benchmark if row['split']=='benchmark_dev' else a.data
            source,target,timing=load_pair(root,row,cfg)
            mask=torch.ones(1,*source.shape[:1],*source.shape[-2:])
            if 'mask' in row:
                mask_row=dict(row,source=row['mask'],target=row['mask'])
                masks,_,_=load_pair(root,mask_row,cfg)
                if masks.shape!=source.shape:
                    raise ValueError('Official mask/source geometry differs; cannot silently remap mask')
                mask=(masks[:,:1].permute(1,0,2,3).float()/255>.5).float()
        except Exception as exc:
            bad.append(dict(id=row['id'],error=repr(exc)))
            print('[editing skip] '+json.dumps(bad[-1]),flush=True)
            continue
        if text is None:
            text=load_text(a.model,device)
            vae=load_vae(a.model,device)
        top,left,height,width=timing['spatial']['valid_rect']
        raw_a=source[:,:,top:top+height,left:left+width].float()/255
        raw_b=target[:,:,top:top+height,left:left+width].float()/255
        quality=dict(source_target_l1=float((raw_b-raw_a).abs().mean()),
            changed_pixel_fraction=float(((raw_b-raw_a).abs().amax(1)>8/255).float().mean()),
            source_frame_delta=float((raw_a[1:]-raw_a[:-1]).abs().mean()),
            target_frame_delta=float((raw_b[1:]-raw_b[:-1]).abs().mean()),
            interpretation='Pixel diagnostics, not guaranteed edit/motion/geometry correctness; report only.')
        del raw_a,raw_b
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
            prompts=text([row['instruction'],cfg['negative_prompt'],''],device)
            src=source.permute(1,0,2,3).to(device).float()/127.5-1
            dst=target.permute(1,0,2,3).to(device).float()/127.5-1
            context=pack_context(vae,src,mask.to(device))
            clean=vae.encode([dst])[0]
        store.save(name,dict(source=source,target=target,mask=mask.byte(),
            clean=clean.cpu().bfloat16(),context=context.cpu().bfloat16(),
            text=prompts[0].cpu().bfloat16(),negative=prompts[1].cpu().bfloat16(),
            empty=prompts[2].cpu().bfloat16(),timing=timing,row=row,quality=quality),identity)
        reports.append(dict(row,file=name,spatial=timing['spatial'],original_fps=timing['original_fps'],quality=quality)); completed+=1
        progress(store,f'cache_rank{rank}',dict(completed=completed,phase='native_cache',rank=rank,
            DI_throughput=completed*cfg['frames']/max(1e-6,time.monotonic()-started),throughput_unit='decoded target frames/s per rank'))
    del text,vae
    all_reports=sum(gather(reports,world),[]); all_bad=sum(gather(bad,world),[])
    if rank==0:
        bank=dict(identity=identity,config=cfg,model_signature=active[0],manifest=manifest,
                  records=sorted(all_reports,key=lambda r:r['id']),skipped=all_bad)
        store.write_json('bank.json',bank)
        counts={s:sum(r['split']==s for r in all_reports) for s in ('train','dev','benchmark_dev')}
        store.write_json('coverage.json',dict(cached=counts,failed=len(all_bad),
            benchmark_requested=len(manifest['benchmark']),test_cached=False,
            source_sizes=dict(Counter(str(r['spatial']['original']) for r in all_reports)),
            tensor_sizes=dict(Counter(str(r['spatial']['canvas']) for r in all_reports)),
            fps_pairs=dict(Counter(str(r['original_fps']) for r in all_reports))))
        print('[cache coverage] '+json.dumps(counts)+'; skipped='+str(len(all_bad)),flush=True)
    barrier()
    if not any(r['split']=='train' for r in all_reports): raise ValueError('No decodable training pairs')
    if not any(r['split']=='dev' for r in all_reports): raise ValueError('No decodable development pairs')
    if dist.is_initialized(): dist.destroy_process_group()


def cache_teacher(a,cfg):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from utils.editing_train import distributed,barrier,gather,progress,load_teacher,load_encoder
    from utils.scene_run import ArtifactStore
    from utils.editing_data import file_hash
    from data.token_utils import strip_special_tokens
    device,rank,world=distributed(); store=ArtifactStore(a.output,a.root)
    bank=store.json('bank.json',required=True)
    signature=dict(rae=file_hash(a.rae),encoder=file_hash(a.encoder)) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    active=[x for x in gather(signature,world) if x is not None]
    if any(x!=active[0] for x in active): raise ValueError('Nodes have different teacher weights')
    identity=digest(dict(bank=bank['identity'],teacher=active[0],schema='edited-target-scene-rae-v1'))
    rae=encoder=None; reports=[]; started=time.monotonic(); completed=0
    for row in [r for r in bank['records'] if r['split'] in ('train','dev')][rank::world]:
        name='rae/'+row['id']+'.pt'
        old=store.load(name,identity)
        if old is None:
            if rae is None:
                rae=load_teacher(a.rae,device); encoder=load_encoder(a.encoder,device)
            item=store.load(row['file'],bank['identity'],True)
            ids=torch.linspace(0,cfg['frames']-1,cfg['rae_frames']).round().long()
            from utils.editing_data import rae_canvas
            square,rect=rae_canvas(item['target'][ids].to(device).float()/255,item['timing']['spatial']['valid_rect'])
            frames=square[None]
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
                tokens,start=encoder(frames)
                z=rae.encode(strip_special_tokens(tokens,start),frames)
            if z.shape!=(1,cfg['rae_frames'],18,18,256) or not torch.isfinite(z).all():
                raise ValueError('Production Scene C256 teacher contract differs')
            store.save(name,dict(z=z[0].cpu().bfloat16(),valid_rect=rect),identity)
            if row['split']=='dev':
                from utils.editing_train import write_video,pixel_report
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
                    decoded=rae.decode(z)[0].permute(0,3,1,2).float().clamp(0,1)
                src,_=rae_canvas(item['source'][ids].to(device).float()/255,item['timing']['spatial']['valid_rect'])
                y,x,h,w=rect
                values=[(v[:,:,y:y+h,x:x+w]*255).round().byte().cpu() for v in (src,square,decoded)]
                movie='teacher_previews/'+row['id']+'.mp4'
                write_video(store.local/movie,*values,cfg['fps']*(cfg['rae_frames']-1)/(cfg['frames']-1),
                    labels=['SOURCE','EDITED TARGET','FROZEN AE RECONSTRUCTION'])
                store.publish([movie])
                store.write_json('teacher_previews/'+row['id']+'.json',pixel_report(*values))
            completed+=1
            progress(store,f'teacher_rank{rank}',dict(phase='edited_target_rae_cache',completed=completed,rank=rank,
                DI_throughput=completed*cfg['rae_frames']/max(1e-6,time.monotonic()-started),throughput_unit='teacher frames/s per rank'))
        reports.append(dict(id=row['id'],file=name))
    rows=sum(gather(reports,world),[])
    if rank==0: store.write_json('teacher_bank.json',dict(identity=identity,native_identity=bank['identity'],signature=active[0],records=rows))
    barrier()
    if dist.is_initialized(): dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('index','cache','teacher'),required=True)
    for flag in ('config','data','model','benchmark','output'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--source'); p.add_argument('--root',action='append',default=[])
    p.add_argument('--rae'); p.add_argument('--encoder')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    {'index':run_index,'cache':cache,'teacher':cache_teacher}[a.stage](a,cfg)


if __name__=='__main__': main()
