"""Complete-cohort public colorization report, with dual-replica media repair."""
from collections import Counter
import csv
from html import escape
import json
import time

import numpy as np

from utils.editing_public_data import ARMS
from utils.editing_paper_metrics import aggregate
from utils.editing_publication import reconcile_case
from utils.native_audit_io import seal, recover_case
from edit_reference_video import crop_video
from refine_vace_edit_state import write_panels

LIMITS = [
    'Official VACE-Benchmark V2V/gray inputs and output descriptions; custom 17-frame/8fps window, not its official leaderboard.',
    'This tests gray-to-color appearance transfer, not arbitrary text replacement, object localization, or unseen-view editing.',
    'Original RGB is evaluated only after generation; all arms share a model-generated anchor, never a ground-truth color anchor.',
    'Native is frozen VACE with the same generated reference image. SFT/RGB/RAE additionally use the same propagated RGB video context.',
    'RGB and RAE are fixed Stage65 trained systems; SFT is the completed Stage69 reference-SFT. No public benchmark finetuning or checkpoint selection.',
    'Prior adaptation/development/evaluation sources are screened by saved IDs and visual fingerprints. Foundation pretraining overlap and scene disjointness are unknown.',
    'Colorization is ambiguous. RGB reference similarity is not proof of semantic prompt correctness; grayscale structure and motion diagnostics accompany it.',
    'DIS supplies method correspondence; independent Farneback supplies approximate evaluation correspondence, not ground-truth geometry.',
    'All-source means and matched complete-case means are separate. Missing outputs and unavailable metrics remain visible; no favorable-case selection.',
    'Sampling runtime includes feedback, decoding and intermediate publication; shared preprocessing and model setup are separate. Arms are not wall-clock matched.',
]


def export_tables(out,tables):
    files=[]
    for label,table in tables.items():
        (out.local/'tables').mkdir(parents=True,exist_ok=True)
        name='tables/'+label+'.csv'
        with (out.local/name).open('w',encoding='utf8',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(['arm','metric','mean','source_groups'])
            for arm,values in table['means'].items():
                for key,value in values.items(): writer.writerow([arm,key,value['mean'],value['source_groups']])
        files.append(name)
        columns=['target_psnr','target_ssim','target_lpips','target_lab_ab_l1','source_gray_ssim']
        lines=[r'\begin{table}[t]',r'\centering',
            r'\caption{VACE-Benchmark gray-to-color transfer, custom 17-frame protocol. '
            r'Paired RGB similarity, chroma error and source-gray structure; see coverage in the report.}',
            r'\begin{tabular}{lrrrrrr}',r'\toprule',
            r'Method & $N$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & AB$\downarrow$ & Gray$\uparrow$ \\',r'\midrule']
        for arm in (*ARMS,'copy_gray'):
            if arm not in table['means']: continue
            values=table['means'][arm]
            entries=[arm.replace('_',r'\_'),str(values.get('target_psnr',{}).get('source_groups',0))]
            entries += ['--' if key not in values else f"{values[key]['mean']:.4f}" for key in columns]
            lines.append(' & '.join(entries)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        name='tables/'+label+'.tex'; (out.local/name).write_text('\n'.join(lines),encoding='utf8'); files.append(name)
    out.publish(files)


def summarize(a,cfg,out):
    from audit_public_editing import checked, cohort
    m=checked(a,cfg,out); selected=cohort(out); sync=[]; failures=[]; coverage={}; runtime={}
    started=time.monotonic()
    def reconcile(receipt):
        try:
            roots=reconcile_case(out,receipt,m['identity'])
            sync.append(dict(receipt=receipt,ok=all(r['ok'] for r in roots),roots=roots))
        except Exception as exc: sync.append(dict(receipt=receipt,ok=False,error=repr(exc)))
        if len(sync)%8==0:
            print('[public media] '+json.dumps(dict(verified=sum(r['ok'] for r in sync),
                DI_throughput=sum(r['ok'] for r in sync)*cfg['frames']/max(time.monotonic()-started,1e-6),
                throughput_unit='verified media frames/s including dual readback')),flush=True)
    expected={(r['id'],s) for r in selected for s in cfg['seeds']}
    for arm in ARMS:
        phase=(out.json('phases/render_'+arm+'.json') or {}).get('records',[])
        complete={(r['id'],r['seed_base']):r for r in phase if r.get('status')=='completed'}
        missing=sorted(expected-set(complete)); unexpected=sorted(set(complete)-expected)
        coverage[arm]=dict(expected=len(expected),completed=len(expected&set(complete)),missing=missing,unexpected=unexpected)
        if missing or unexpected: failures.append(dict(phase='coverage',arm=arm,missing=missing,unexpected=unexpected))
        failures.extend(r for r in phase if r.get('status')!='completed')
        for row in complete.values():
            reconcile(row['receipt'])
            for event in row.get('feedback_events',[]):
                if event.get('applied'):
                    reconcile(f'render/{arm}/seed{row["seed_base"]}/{row["id"]}/feedback/step_{int(event["step_index"]):03d}.json')
        runtime[arm]={}
        for key in ('sampling_seconds','setup_seconds','peak_allocated_gib'):
            values=[r[key] for r in complete.values() if r.get(key) is not None]
            runtime[arm][key]=dict(mean=float(np.mean(values)),max=float(max(values)),n=len(values)) if values else None
        print('[public coverage] '+arm+' '+str(coverage[arm]),flush=True)
    evaluation=(out.json('phases/evaluate.json') or {}).get('records',[])
    good=[r for r in evaluation if r.get('status')=='completed']
    evaluated={(r['arm'],r['id'],r['seed_base']) for r in good}
    missing=[dict(arm=arm,id=key,seed=seed) for arm in ARMS for key,seed in sorted(expected)
             if (arm,key,seed) not in evaluated]
    if missing: failures.append(dict(phase='evaluate',missing=missing))
    failures.extend(r for r in evaluation if r.get('status')!='completed')
    matched={(key,seed) for key,seed in expected if all((arm,key,seed) in evaluated for arm in ARMS)}
    paired=[r for r in good if (r['id'],r['seed_base']) in matched]
    baseline=[out.json('copy_gray/'+r['id']+'.json') for r in selected]
    baseline=[r for r in baseline if r]
    tables=dict(matched=aggregate(paired,ARMS,10000),observed=aggregate(good,ARMS,10000),
        copy_gray=aggregate(baseline,[],10000))
    for domain in ('real','synthetic'):
        tables[domain]=aggregate([r for r in paired if r['domain']==domain],ARMS,10000)
    export_tables(out,tables)
    page=['<!doctype html><meta charset="utf-8"><title>Public colorization transfer</title>',
        '<style>body{font:16px Arial;max-width:1700px;margin:24px auto}video{width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>VACE-Benchmark: gray-to-color transfer</h1><p>All selected cases; shared generated anchor. '
        'Original RGB is an evaluation label, never an inference condition.</p>']
    for row in selected:
        page.append('<h2>'+escape(row['id']+' — '+row['instruction'])+'</h2>')
        reconcile('anchors/'+row['id']+'/receipt.json')
        for seed in cfg['seeds']:
            try:
                payloads=[out.load(f'render/{arm}/seed{seed}/{row["id"]}/result.pt',m['identity'],True) for arm in ARMS]
                if len({p['input_identity'] for p in payloads})!=1: raise ValueError('Four-arm inputs differ')
                source=out.load(row['source_file'],m['identity'],True)
                rect=source['spatial']['valid_rect']; folder=f'comparisons/seed{seed}/{row["id"]}'
                receipt=folder+'/receipt.json'; movie=folder+'/comparison.mp4'
                generations=[p['generation_identity'] for p in payloads]
                prior=recover_case(out.local,receipt,m['identity'],out.roots)
                if prior is None or prior['metadata'].get('generations')!=generations:
                    # No target decode here: comparisons remain source + four final outputs.
                    write_panels(out.local/movie,[crop_video(source['source'].float()/255,rect)]+[
                        crop_video(p['generated'],rect) for p in payloads],['GRAY SOURCE',*ARMS],cfg['fps'])
                    seal(out.local,receipt,m['identity'],[movie],dict(generations=generations))
                out.publish([movie,receipt]); reconcile(receipt)
                page.append(f'<p>Seed {seed}</p><video controls preload="none" src="{movie}"></video>')
            except Exception as exc:
                failures.append(dict(id=row['id'],phase='comparison',seed=seed,error=repr(exc)))
                page.append('<p>'+escape('Comparison unavailable: '+repr(exc))+'</p>')
    # Expose upstream errors instead of reporting only downstream missing files.
    upstream=[]
    for phase in ('sources','text','anchors','shared','cache','prepare'):
        records=(out.json('phases/'+phase+'.json') or {}).get('records',[])
        upstream += [dict(r,phase=phase) for r in records if r.get('status')!='completed']
    metric_names=sorted({key for r in good for key in r['metrics']})
    metric_coverage={k:sum(r['metrics'].get(k) is not None for r in good) for k in metric_names}
    bad=[r for r in sync if not r['ok']]
    report=dict(complete=not failures and not bad,identity=m['identity'],protocol=m['protocol'],
        benchmark_audit=m['audit'],cohort=out.json('cohort.json'),coverage=coverage,
        evaluated=len(evaluated),evaluation_expected=len(expected)*len(ARMS),
        matched_case_seeds=len(matched),tables=tables,runtime=runtime,metric_coverage=metric_coverage,
        lpips_status=sorted({json.dumps(r['lpips'],sort_keys=True) for r in good}),
        failures=failures,upstream_failures=upstream,media_failures=bad,
        failure_causes=dict(Counter(r.get('error',r.get('phase','unknown')) for r in upstream+failures)),
        limitations=LIMITS,trained_on_public_benchmark=False)
    out.write_json('summary.json',report)
    out.write_json('media_sync.json',dict(checked=len(sync),failed=len(bad),records=sync))
    page.append('<pre>'+escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(page),encoding='utf8'); out.publish(['report.html'])
    print('[public summary] '+json.dumps(dict(complete=report['complete'],coverage=coverage,failures=report['failure_causes'])),flush=True)
    if not report['complete']: raise RuntimeError('Public suite incomplete; completed cases retained; inspect summary.json')
