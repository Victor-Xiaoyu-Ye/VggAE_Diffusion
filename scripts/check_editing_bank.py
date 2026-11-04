"""Check/recover metadata for an explicitly reused editing bank, without writes."""
import argparse
import json
from utils.scene_run import ArtifactStore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--root',action='append',default=[])
    p.add_argument('--require-teacher',action='store_true')
    a=p.parse_args();store=ArtifactStore(a.bank,a.root);bank=store.json('bank.json',True)
    counts={s:sum(r['split']==s for r in bank['records']) for s in ('train','dev','benchmark_dev')}
    if not counts['train'] or not counts['dev']:raise ValueError('Reused bank has no train/dev records')
    if a.require_teacher:
        teacher=store.json('teacher_bank.json',True)
        if teacher['native_identity']!=bank['identity']:raise ValueError('Teacher/native bank identities disagree')
        available={r['id'] for r in teacher['records']}
        if any(r['id'] not in available for r in bank['records'] if r['split'] in ('train','dev')):
            raise ValueError('Reused teacher bank is incomplete; restore it or prepare a new independent bank')
    print('[editing bank reuse] '+json.dumps(dict(identity=bank['identity'],counts=counts,
        teacher_required=a.require_teacher,remote_mutations=False)),flush=True)


if __name__=='__main__':main()
