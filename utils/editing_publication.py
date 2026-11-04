"""Reconcile sealed preview media across BOTH outputs, including resumed cases.

Only small evaluation JSON and receipt-declared media are read. Never walk or
copy model checkpoints, native caches, or unrelated experiment directories.
"""
import hashlib
import json
import time
from pathlib import Path
from scripts.window_run_io import child
from utils.moxing_io import is_remote_path, open_file, read_text, copy_file
from utils.native_audit_io import exists, safe_relative, recover_case


def evaluation_names(store):
    names=set(); errors=[]
    for root in [str(store.local), *store.roots]:
        folder=child(root,'evaluation')
        try:
            if not exists(folder): continue
            if is_remote_path(folder):
                import moxing
                entries=moxing.file.list_directory(folder,recursive=False)
                for entry in entries:
                    # MoXing versions return either relative names or full URIs.
                    name=entry.rsplit('/',1)[-1]
                    if name.endswith('.json'): names.add('evaluation/'+safe_relative(name))
            else:
                names.update('evaluation/'+p.name for p in Path(folder).glob('*.json'))
        except Exception as exc:
            errors.append(dict(root=root,error=repr(exc)))
    return sorted(names),errors


def file_matches(path, expected):
    if not exists(path): return False
    h=hashlib.sha256(); size=0
    with open_file(path,'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            size+=len(block); h.update(block)
    return size==expected['size'] and h.hexdigest()==expected['sha256']


def reconcile_case(store, name, contract, attempts=3, delay=1.):
    """Verified read-back; receipt written LAST, after this root's payloads."""
    receipt=recover_case(store.local,name,contract,store.roots)
    if receipt is None: raise FileNotFoundError(name)
    results=[]
    for root in store.roots:
        for attempt in range(attempts):
            try:
                repaired=[]
                for relative,expected in receipt['files'].items():
                    destination=child(root,safe_relative(relative))
                    if not file_matches(destination,expected):
                        copy_file(str(store.local/relative),destination)
                        if not file_matches(destination,expected):
                            raise IOError('Published media checksum mismatch: '+relative)
                        repaired.append(relative)
                destination=child(root,name)
                if not exists(destination) or json.loads(read_text(destination))!=receipt:
                    copy_file(str(store.local/name),destination)
                if json.loads(read_text(destination))!=receipt:
                    raise IOError('Published receipt differs: '+name)
                results.append(dict(root=root,ok=True,repaired=repaired)); break
            except Exception as exc:
                if attempt+1==attempts:
                    results.append(dict(root=root,ok=False,error=repr(exc)))
                elif delay: time.sleep(delay)
    return results


def reconcile_evaluations(store):
    names,listing_errors=evaluation_names(store)
    evaluations={}; cases=[]
    for name in names:
        value=store.json(name,True); evaluations[name]=value
        for row in value['cases']:
            if 'error' in row: continue
            tag=row['variant']+('_source_region' if value.get('protocol')=='text_plus_source_box_diagnostic' else '')
            receipt_name=f"samples/s{row['step']:06d}/{tag}/{safe_relative(row['id'])}/receipt.json"
            try:
                receipt=store.json(receipt_name,True)
                roots=reconcile_case(store,receipt_name,receipt['contract_hash'])
                cases.append(dict(receipt=receipt_name,roots=roots,ok=all(r['ok'] for r in roots)))
            except Exception as exc:
                cases.append(dict(receipt=receipt_name,ok=False,error=repr(exc)))
            if len(cases)%24==0:
                print(f'[editing media reconcile] checked={len(cases)}',flush=True)
        # Evaluation metadata travels with the media, even after recovery.
        store.write_json(name,value)
    report=dict(status='synced' if cases and not listing_errors and all(c['ok'] for c in cases) else 'incomplete',
                checked=len(cases),failed=sum(not c['ok'] for c in cases),listing_errors=listing_errors,cases=cases)
    store.write_json('media_sync.json',report)
    return report,evaluations
