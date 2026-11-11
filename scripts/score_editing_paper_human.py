"""Aggregate real Stage69 blinded ratings; no automatic or invented judgments."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

AXES=('instruction','source_preservation','temporal_quality','overall_quality')


def score(key,submissions,seed=42,bootstrap=2000):
    trials={t['trial']:t for t in key['trials']}
    seen=set();raters=set();groups={};counts={}
    for submission in submissions:
        rater=str(submission.get('rater','')).strip()
        if not rater:raise ValueError('Missing independent rater identifier')
        raters.add(rater)
        for row in submission['ratings']:
            token=row['trial']
            if token not in trials:raise ValueError('Unknown blinded trial '+token)
            if (rater,token) in seen:raise ValueError('Duplicate submission for rater/trial')
            seen.add((rater,token));trial=trials[token]
            if 'rae_full' not in (trial['a'],trial['b']):raise ValueError('Unexpected comparison key')
            control=trial['b'] if trial['a']=='rae_full' else trial['a']
            for axis in AXES:
                value=row['ratings'].get(axis)
                if value not in (None,'A','B','tie','neither'):raise ValueError('Invalid human choice')
                count=counts.setdefault((control,axis),dict(A=0,B=0,tie=0,neither=0,missing=0))
                count['missing' if value is None else value]+=1
                if value is None or value=='neither':continue
                winner=trial['a'] if value=='A' else trial['b']
                pref=.5 if value=='tie' else float(winner=='rae_full')
                groups.setdefault((control,axis),{}).setdefault(trial['group'],[]).append(pref)
    result={}
    for (control,axis),count in counts.items():
        source_scores=np.array([np.mean(v) for v in groups.get((control,axis),{}).values()])
        report=dict(counts=count,source_groups=len(source_scores))
        if len(source_scores):
            rng=np.random.default_rng(seed)
            means=source_scores[rng.integers(0,len(source_scores),(bootstrap,len(source_scores)))].mean(1)
            report.update(rae_preference=float(source_scores.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist())
        result.setdefault(control,{})[axis]=report
    return dict(status='rated' if seen else 'no_ratings',raters=len(raters),submitted_trial_rows=len(seen),
        expected_trial_rows_per_rater=len(trials),results=result,
        missing_trial_rows=[dict(rater=r,trial=t) for r in sorted(raters) for t in trials if (r,t) not in seen],
        aggregation='Average actual rater preferences per source then source bootstrap; tie=0.5; neither excluded and reported, not counted as wins',
        caveat='Rater independence/blinding must be enforced during collection; this script cannot certify it.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--key',required=True)
    p.add_argument('--ratings',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();key=json.loads(Path(a.key).read_text(encoding='utf8'))
    submissions=[json.loads(Path(path).read_text(encoding='utf8')) for path in a.ratings]
    result=score(key,submissions);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(raters=result['raters'],submitted_trial_rows=result['submitted_trial_rows'],output=str(path))))


if __name__=='__main__':main()
