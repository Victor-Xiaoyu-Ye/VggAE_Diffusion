"""Three node-local T5 workers, resumable immutable shards and lazy-reader index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.window_run_io import child
from utils.moxing_io import read_bytes, copy_file
from utils.native_audit_io import exists
from utils.file_signature import sampled_file_signature
from utils.window_training import digest
from utils.training import ThroughputMeter


def publish(root, staging, name, payload, binary=False):
    path=Path(staging)/name;path.parent.mkdir(parents=True,exist_ok=True)
    if binary: torch.save(payload,path)
    else: path.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    copy_file(str(path),child(root,name));path.unlink()
    return checksum


def main():
    p=argparse.ArgumentParser()
    for name in ('csv','annotation_root','output','staging','wan_ckpt'):p.add_argument('--'+name,required=True)
    p.add_argument('--merge',action='store_true');p.add_argument('--nodes',type=int,default=3)
    p.add_argument('--text_len',type=int,default=256);p.add_argument('--shard_size',type=int,default=32)
    p.add_argument('--max_missing_fraction',type=float,default=.05)
    a=p.parse_args();rank=int(os.environ.get('NODE_RANK','0'))
    if a.nodes < 1 or not 0 <= rank < a.nodes or a.text_len < 1 or a.shard_size < 1:
        p.error('invalid node/rank/text/shard dimensions')
    if not 0 <= a.max_missing_fraction < 1:
        p.error('max_missing_fraction must be in [0,1)')
    with open(a.csv,encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('text input must contain unique nonempty rows')
    tokenizer=Path(a.wan_ckpt)/'google/umt5-xxl'
    tokenizer_files={str(path.relative_to(tokenizer)):hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(tokenizer.rglob('*')) if path.is_file()}
    if not tokenizer_files:raise FileNotFoundError(f'empty tokenizer directory: {tokenizer}')
    identity=dict(schema='fullhq-text-source-v1',csv_sha256=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
        annotation_root=a.annotation_root,text_len=a.text_len,nodes=a.nodes,shard_size=a.shard_size,
        tokenizer=tokenizer_files,
        t5=sampled_file_signature(str(Path(a.wan_ckpt)/'models_t5_umt5-xxl-enc-bf16.pth')))
    signature=digest(identity)
    if a.merge:
        index={};files={};missing=0
        for node in range(a.nodes):
            part=json.loads(read_bytes(child(a.output,f'node{node}/complete.json')))
            if part['identity']!=identity:raise ValueError('text worker identity differs')
            if set(index)&set(part['index']):raise ValueError('duplicate text IDs')
            index.update(part['index']);files.update(part['files']);missing+=part['missing']
        if set(index)!={r['id'] for r in rows}:raise ValueError('text coverage differs from selected data')
        if missing/len(rows)>a.max_missing_fraction:raise ValueError('too many missing captions; inspect text receipts')
        zero=json.loads(read_bytes(child(a.output,'empty_receipt.json')))
        if zero['signature']!=signature:raise ValueError('empty prompt identity changed')
        files['empty_prompt.pt']=zero['sha256']
        manifest=dict(schema='fullhq-text-index-v1',identity=identity,index=index,files=files,text_len=a.text_len,
                      missing_captions=missing,missing_policy='explicit empty prompt; keep image-only training examples')
        checksum=publish(a.output,a.staging,'index.json',manifest)
        publish(a.output,a.staging,'_SUCCESS',dict(schema='fullhq-text-v1',index_sha256=checksum,num_videos=len(index)))
        print(f'[fullHQ text] merged {len(index)} IDs, {missing} empty captions',flush=True);return
    own=rows[rank::a.nodes];index={};files={};missing=0;encoder=None
    throughput_meter=ThroughputMeter()
    for start in range(0,len(own),a.shard_size):
        group=own[start:start+a.shard_size];stem=f'node{rank}/shard{start//a.shard_size:06d}'
        receipt_path=child(a.output,stem+'.json')
        if exists(receipt_path):
            receipt=json.loads(read_bytes(receipt_path))
            if receipt['signature']!=signature or receipt['ids']!=[r['id'] for r in group]:raise ValueError('text shard resume mismatch')
            for name in receipt['files']:
                if not exists(child(a.output,name)):raise FileNotFoundError(name)
        else:
            if encoder is None:
                from precompute_wan_text_embeddings import _load_wan_module
                from utils.device import get_device,get_device_name
                device=get_device(0)
                if get_device_name()!='npu':raise RuntimeError('full-HQ text stage requires NPU')
                torch.npu.set_device(0)
                encoder=_load_wan_module('t5').T5EncoderModel(text_len=a.text_len,dtype=torch.bfloat16,device=device,
                    checkpoint_path=str(Path(a.wan_ckpt)/'models_t5_umt5-xxl-enc-bf16.pth'),
                    tokenizer_path=str(Path(a.wan_ckpt)/'google/umt5-xxl'))
                if rank==0:
                    with torch.no_grad(): empty=encoder([''],device)[0].cpu().half()
                    sha=publish(a.output,a.staging,'empty_prompt.pt',empty,True)
                    publish(a.output,a.staging,'empty_receipt.json',dict(signature=signature,sha256=sha))
            def fetch(row):
                from data.annotation_index import _derive_group
                url=child(a.annotation_root,f"{_derive_group(row['video path'])}/{row['id']}/caption.json")
                error=''
                for attempt in range(3):
                    try:
                        cap=json.loads(read_bytes(url)).get('SceneDescription','')
                        if not isinstance(cap,str) or not cap.strip():return '', 'empty SceneDescription'
                        return cap.strip(),''
                    except Exception as exc:error=repr(exc);time.sleep(.2*(attempt+1))
                return '',error
            with ThreadPoolExecutor(max_workers=8) as pool:captions=list(pool.map(fetch,group))
            tensors={};issues=[]
            valid=[(r['id'],cap) for r,(cap,error) in zip(group,captions) if cap]
            with torch.no_grad():
                for offset in range(0,len(valid),2):
                    batch=valid[offset:offset+2]
                    values=encoder([t for _,t in batch],device)
                    for (vid,_),value in zip(batch,values):
                        if not torch.isfinite(value).all():raise RuntimeError('nonfinite T5 output')
                        tensors[vid]=value.cpu().half()
                        throughput_meter.update(value.shape[0])
            for r,(cap,error) in zip(group,captions):
                if not cap:issues.append(dict(video_id=r['id'],error=error))
            file=stem+'.pt';filehash=publish(a.output,a.staging,file,
                dict(embeddings=tensors,captions={vid:cap for vid,cap in valid}),True)
            receipt=dict(signature=signature,ids=[r['id'] for r in group],
                index={r['id']:file if r['id'] in tensors else None for r in group},files={file:filehash},
                missing=len(issues),issues=issues)
            publish(a.output,a.staging,stem+'.json',receipt)
        index.update(receipt['index']);files.update(receipt['files']);missing+=receipt['missing']
        done=start+len(group)
        print(f'[fullHQ text] node{rank} {done}/{len(own)} missing={missing} '
              f'DI_throughput: {throughput_meter.format()} (new text tokens; one active NPU/node)',flush=True)
        if done>=128 and missing/done>a.max_missing_fraction:raise RuntimeError('caption failure fraction too high')
    publish(a.output,a.staging,f'node{rank}/complete.json',dict(identity=identity,index=index,files=files,missing=missing))


if __name__=='__main__':main()
