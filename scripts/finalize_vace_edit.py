"""Repair preview replicas and summarize matched protocols; no model loading."""
import argparse
import json
import os
from pathlib import Path
from utils.scene_run import ArtifactStore
from scripts.window_run_io import child
from utils.editing_publication import reconcile_evaluations


METRICS=('target_psnr','source_change','target_change','output_frame_delta','edit_region_target_l1','keep_region_source_l1',
         'keep_region_target_l1','edit_region_output_change','edit_region_target_change',
         'keep_temporal_delta_error','copy_source_target_psnr')


def paired_summary(all_evaluations):
    comparisons=[]
    for baseline,method in (('native','sft'),('sft','rae'),('sft','rae_detached'),('region_sft','rae_balanced')):
        left=all_evaluations.get(baseline,{})
        right=all_evaluations.get(method,{})
        for name,b in sorted(right.items()):
            suffix='_source_region' if b.get('protocol')=='text_plus_source_box_diagnostic' else ''
            a=left.get('evaluation/s000000_native'+suffix+'.json' if baseline=='native' else name)
            if a is None:
                comparisons.append(dict(baseline=baseline,method=method,evaluation=name,status='missing_baseline'));continue
            ar={r['id']:r for r in a['cases'] if 'error' not in r and r.get('split')=='dev'}
            br={r['id']:r for r in b['cases'] if 'error' not in r and r.get('split')=='dev'}
            ids=sorted(ar.keys() & br.keys())
            mismatched=[i for i in ids if any(ar[i].get(k)!=br[i].get(k) for k in ('seed','prompt','spatial','control_identity'))]
            ids=[i for i in ids if i not in mismatched]
            metrics={}
            for key in METRICS:
                valid=[i for i in ids if isinstance(ar[i].get(key),(float,int)) and isinstance(br[i].get(key),(float,int))]
                if valid:
                    metrics[key]=dict(count=len(valid),baseline=sum(ar[i][key] for i in valid)/len(valid),
                        method=sum(br[i][key] for i in valid)/len(valid),
                        delta_method_minus_baseline=sum(br[i][key]-ar[i][key] for i in valid)/len(valid))
            comparisons.append(dict(baseline=baseline,method=method,evaluation=name,protocol=b.get('protocol','text_only'),
                paired=len(ids),ids=ids,missing_method=sorted(ar.keys()-br.keys()),missing_baseline=sorted(br.keys()-ar.keys()),
                mismatched_conditions=mismatched,baseline_failures=a.get('failed',0),method_failures=b.get('failed',0),
                metrics=metrics,status='matched' if ids else 'no_comparable_cases'))
    return dict(comparisons=comparisons,
        caveat='Pixel/change-support diagnostics are not instruction adherence, geometry accuracy or a paper-level benchmark. No automatic best checkpoint selected.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--root',action='append',default=[])
    p.add_argument('--arms',nargs='+',default=['native','sft','rae'])
    a=p.parse_args()
    # All three coordinators invoke us; only node0 does the lightweight sweep.
    if int(os.environ.get('NODE_RANK','0'))!=0:return
    result={};errors=[];reports={}
    for arm in a.arms:
        try:
            store=ArtifactStore(Path(a.output)/arm,[child(r,arm) for r in a.root])
            report,evaluations=reconcile_evaluations(store)
            reports[arm]={k:v for k,v in report.items() if k!='cases'};result[arm]=evaluations
            if report['status']!='synced':errors.append(arm)
        except Exception as exc:
            reports[arm]=dict(status='failed',error=repr(exc));errors.append(arm)
    out=ArtifactStore(a.output,a.root)
    out.write_json('comparison_summary.json',dict(paired_summary(result),media=reports))
    print('[editing finalization] '+json.dumps(reports),flush=True)
    if errors:raise SystemExit('Media sync incomplete for: '+', '.join(errors)+'; rerun EDIT_STAGE=sync, no training required.')


if __name__=='__main__':main()
