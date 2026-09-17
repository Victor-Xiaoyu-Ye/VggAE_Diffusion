"""All metadata rows, deterministic heldout exclusion, explicit invalid-row ledger."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.window_run_io import atomic


def prepare(metadata, protocol, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=hashlib.sha256(Path(metadata).read_bytes()).hexdigest()
    spec=json.loads(Path(protocol).read_text())
    if source != spec['metadata_sha256']: raise ValueError('HQ metadata checksum changed')
    heldout={v for k,values in spec['splits'].items() if k.startswith(('eval_', 'test_')) for v in values}
    seen=set();train=[];evaluation=[];invalid=[]
    eval_ids=set(spec['splits']['eval_street_128'])
    with open(metadata,encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        for index,row in enumerate(reader):
            try:
                vid=row['id'].strip()
                if not vid or vid in seen: raise ValueError('empty/duplicate video ID')
                seen.add(vid)
                fps=float(row['fps']);frames=int(row['num frames'])
                path=row['video path']
                if not math.isfinite(fps) or fps<8 or frames<max(9,round(fps)+1) or not path.endswith('.mp4') or '..' in path.split('/'):
                    raise ValueError('invalid video metadata')
                if vid in eval_ids: evaluation.append(row)
                elif vid not in heldout: train.append(row)
            except (ValueError,KeyError) as exc:
                invalid.append(dict(row=index,video_id=row.get('id'),error=str(exc)))
    if len(evaluation)!=len(eval_ids) or not train: raise ValueError('missing reviewed eval IDs or empty training set')
    for name,rows in (('train.csv',train),('eval.csv',evaluation),('text.csv',train+evaluation)):
        temp=out/(name+'.tmp')
        with temp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
        temp.replace(out/name)
    report=dict(schema='fullhq-selection-v1',metadata_sha256=source,
        train_videos=len(train),eval_videos=len(evaluation),reserved_ids=len(heldout),
        invalid=invalid,split='all valid metadata rows minus all existing eval/test IDs; no scene filter',
        source_disjoint_verified=False,
        csv_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('train.csv','eval.csv','text.csv')})
    atomic(out/'selection.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='invalid'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.metadata,a.protocol,a.output)
