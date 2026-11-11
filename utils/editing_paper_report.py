"""Coverage-first paper tables, synchronized media and blinded human packets."""
import csv
import json
import time
from html import escape
from pathlib import Path

import numpy as np

from utils.editing_data import digest
from utils.editing_paper_metrics import aggregate
from utils.editing_publication import reconcile_case
from utils.native_audit_io import seal, recover_case
from edit_reference_video import crop_video
from refine_vace_edit_state import write_panels


METRICS=('target_psnr','target_ssim','target_lpips','dataset_edit_l1',
         'dataset_keep_source_l1','source_tracked_effect_l1','frame_delta_ratio')
LIMITS=[
    'Reserved source-byte groups with source-only visual screening; parent-scene separation is not certified.',
    'Old 24 development cases are not relabeled as independent test data.',
    'Paired targets are dataset-generated references; reference similarity is not semantic instruction correctness.',
    'Evaluation change masks come from source/target differences, not semantic ground truth; never sampling inputs.',
    'Independent Farneback flow is approximate correspondence; static or wrong edits can have low temporal error.',
    'Native means frozen VACE with the same model-generated anchor; native_reference also gets the shared propagated RGB reference.',
    'New anchors use frozen native VACE, unlike the earlier development frontend. All new methods share these anchors.',
    'SFT uses the same Stage65 data, sampler, maximum updates and stopping rule; actual updates and trainable sizes may differ.',
    'Main table uses the first seed only. Ablations use the same preselected subset and seed. Extra-seed robustness is separate.',
    'No VAE/DINO representation control or public external-editor leaderboard is completed by this suite.',
    'Human preference results are pending until independent raters submit the blinded packets.']


def metric_tables(records,cfg,selected):
    first=[r for r in records if r.get('seed_base')==cfg['seed']]
    subset=set(selected['ablation_ids'])
    return dict(main=aggregate([r for r in first if r.get('arm') in cfg['main_arms']],cfg['main_arms']),
        ablations=aggregate([r for r in first if r.get('id') in subset],cfg['main_arms']+cfg['ablation_arms']),
        robustness=aggregate([r for r in records if r.get('id') in subset and r.get('arm') in cfg['second_seed_arms']],cfg['second_seed_arms']))


def export_tables(out,tables):
    files=[]
    for label,table in tables.items():
        folder=out.local/'tables';folder.mkdir(parents=True,exist_ok=True)
        name='tables/'+label+'.csv'
        with (out.local/name).open('w',newline='',encoding='utf8') as stream:
            writer=csv.writer(stream);writer.writerow(['arm','metric','mean','source_groups'])
            for arm,values in table['means'].items():
                for key,value in values.items():writer.writerow([arm,key,value['mean'],value['source_groups']])
        files.append(name)
        columns=('target_psnr','target_ssim','target_lpips','dataset_keep_source_l1','source_tracked_effect_l1')
        lines=['% Measured diagnostics only; see summary.json for coverage and evaluator limits.',
               r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Method & $N$ & PSNR & SSIM & LPIPS & Keep $L_1$ & Track $L_1$ \\',r'\midrule']
        for arm,values in table['means'].items():
            entries=[arm.replace('_',r'\_'),str(values.get('target_psnr',{}).get('source_groups',0))]
            entries.extend('--' if k not in values else f"{values[k]['mean']:.4f}" for k in columns)
            lines.append(' & '.join(entries)+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        name='tables/'+label+'.tex';(out.local/name).write_text('\n'.join(lines),encoding='utf8');files.append(name)
    out.publish(files)


def human_packet(out,m,cfg,selected):
    """Blinded outputs; private assignment key never goes in the public folder."""
    trials=[];private=[];failures=[];files=[];subset=set(selected['ablation_ids'])
    previous=recover_case(out.local,'human/receipt.json',m['identity'],out.roots)
    private_key=out.json('human_private/key.json')
    if previous is not None and previous['metadata'].get('trials')==len(subset)*3 and private_key:
        roots=reconcile_case(out,'human/receipt.json',m['identity'])
        return dict(trials=len(subset)*3,expected=len(subset)*3,failures=[],roots=roots,
            status='awaiting_independent_ratings',private_key='human_private/key.json',packet='human/index.html')
    for row in selected['records']:
        if row['id'] not in subset:continue
        for control in ('native','rgb_full','rae_base'):
            try:
                token=digest(['human',cfg['seed'],row['group'],control])[:16]
                arms=['rae_full',control]
                if int(token[:8],16)%2:arms.reverse()
                payloads=[out.load(f'render/{arm}/seed{cfg["seed"]}/{row["id"]}/result.pt',m['identity'],True) for arm in arms]
                if len({p['input_identity'] for p in payloads})!=1:raise ValueError('Human comparison inputs differ')
                source=out.load(row['source_file'],m['identity'],True)
                rect=source['spatial']['valid_rect'];name='human/videos/'+token+'.mp4'
                write_panels(out.local/name,[crop_video(source['source'].float()/255,rect)]+
                    [crop_video(p['generated'],rect) for p in payloads],['SOURCE','A','B'],cfg['fps'])
                files.append(name)
                trials.append(dict(trial=token,prompt=row['instruction'],video='videos/'+token+'.mp4'))
                private.append(dict(trial=token,id=row['id'],group=row['group'],a=arms[0],b=arms[1],
                    generation_ids=[p['generation_identity'] for p in payloads]))
            except Exception as exc:failures.append(dict(id=row['id'],control=control,error=repr(exc)))
    trials.sort(key=lambda t:digest(['human-order',t['trial']]))
    out.write_json('human_private/key.json',dict(identity=m['identity'],trials=private))
    out.write_json('human/trials.json',dict(trials=trials,axes=['instruction','source_preservation','temporal_quality','overall_quality']))
    page='''<!doctype html><meta charset="utf-8"><title>Blinded video editing review</title>
<style>body{font:16px Arial;max-width:1400px;margin:24px auto}video{width:100%}section{padding:20px;border-bottom:1px solid #aaa}label{margin-right:16px}p{line-height:1.5}</style>
<h1>Video editing comparison</h1><p>Judge A and B against the source and the requested edit. Do not reward a missed edit merely for preserving the source. Judge motion and realism separately. A/B assignments vary between trials.</p>
<label>Rater ID <input id="rater" required></label><button onclick="save()">Download ratings</button><main id="main"></main>
<script>const trials=TRIALS; const axes={instruction:'Instruction fulfillment',source_preservation:'Preservation of unedited content',temporal_quality:'Temporal consistency / natural motion',overall_quality:'Overall visual quality'};
for(const t of trials){let s=document.createElement('section'),h=document.createElement('h3');h.textContent=t.prompt;s.append(h);let v=document.createElement('video');v.controls=true;v.preload='none';v.src=t.video;s.append(v);
for(const [axis,label] of Object.entries(axes)){let p=document.createElement('p');p.append(label+': ');for(const value of ['A','B','tie','neither']){let l=document.createElement('label'),i=document.createElement('input');i.type='radio';i.name=t.trial+'_'+axis;i.value=value;l.append(i,value);p.append(l);}s.append(p);}main.append(s);}
function save(){if(!rater.value.trim()){alert('Enter a rater ID');return;}let ratings=trials.map(t=>({trial:t.trial,ratings:Object.fromEntries(Object.keys(axes).map(a=>[a,document.querySelector('input[name="'+t.trial+'_'+a+'"]:checked')?.value||null]))}));let b=new Blob([JSON.stringify({rater:rater.value.trim(),ratings},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ratings.json';a.click();}
</script>'''.replace('TRIALS',json.dumps(trials,ensure_ascii=True).replace('<','\\u003c'))
    (out.local/'human/index.html').write_text(page,encoding='utf8');files.extend(['human/index.html','human/trials.json'])
    receipt='human/receipt.json';seal(out.local,receipt,m['identity'],files,dict(trials=len(trials)))
    out.publish([*files,receipt]);roots=reconcile_case(out,receipt,m['identity'])
    return dict(trials=len(trials),expected=len(subset)*3,failures=failures,roots=roots,
        status='awaiting_independent_ratings',private_key='human_private/key.json',packet='human/index.html')


def summarize(a,cfg,out):
    from paper_editing_suite import checked,cohort,selected_tasks
    from train_vace_edit_state import arm_store
    m=checked(a,cfg,out);selected=cohort(out);coverage={};failures=[];sync=[];runtime={};started=time.monotonic()
    arms=cfg['main_arms']+cfg['ablation_arms']
    for arm in arms:
        phase=out.json('phases/render_'+arm+'.json') or {}
        rows=phase.get('records',[]);expected={(r['id'],s) for r,s in selected_tasks(cfg,selected,arm)}
        complete={(r['id'],r['seed_base']):r for r in rows if r.get('status')=='completed'}
        missing=sorted(expected-set(complete));extra=sorted(set(complete)-expected)
        coverage[arm]=dict(expected=len(expected),completed=len(expected&set(complete)),missing=missing,unexpected=extra)
        if missing or extra:failures.append(dict(arm=arm,phase='coverage',missing=missing,unexpected=extra))
        failures.extend(r for r in rows if r.get('status')!='completed')
        for row in complete.values():
            receipts=[row['receipt']]
            for event in row.get('feedback_events',[]):
                if event.get('applied'):
                    receipts.append(f'render/{arm}/seed{row["seed_base"]}/{row["id"]}/feedback/step_{int(event["step_index"]):03d}.json')
            for name in receipts:
                try:
                    roots=reconcile_case(out,name,m['identity'])
                    sync.append(dict(receipt=name,ok=all(r['ok'] for r in roots),roots=roots))
                except Exception as exc:sync.append(dict(receipt=name,ok=False,error=repr(exc)))
            if len(sync)%8==0:
                print('[paper media] '+json.dumps(dict(verified=sum(r['ok'] for r in sync),
                    DI_throughput=sum(r['ok'] for r in sync)*cfg['frames']/max(time.monotonic()-started,1e-6),
                    throughput_unit='verified media frames/s including readback IO')),flush=True)
        runtime[arm]={}
        for key in ('sampling_seconds','setup_seconds','peak_allocated_gib','sample_steps',
                    'feedback_decoder_evaluations','feedback_response_evaluations'):
            values=[r[key] for r in complete.values() if r.get(key) is not None]
            runtime[arm][key]=dict(mean=float(np.mean(values)),max=float(max(values)),n=len(values)) if values else None
        print('[paper summary] '+arm+' '+str(coverage[arm]),flush=True)
    evaluation=(out.json('phases/evaluate.json') or {}).get('records',[])
    evaluated={(r['arm'],r['id'],r['seed_base']) for r in evaluation if r.get('status')=='completed'}
    evaluation_missing=[dict(arm=arm,id=r['id'],seed=s) for arm in arms for r,s in selected_tasks(cfg,selected,arm)
                        if (arm,r['id'],s) not in evaluated]
    if evaluation_missing:failures.append(dict(phase='evaluation',missing=evaluation_missing))
    tables=metric_tables(evaluation,cfg,selected);export_tables(out,tables)
    metric_coverage={key:sum(r.get('status')=='completed' and r.get('metrics',{}).get(key) is not None for r in evaluation) for key in METRICS}
    # Fixed source-flow threshold, not output-dependent winning-case strata.
    strata={}
    for label,low in (('source_flow_below_1px',True),('source_flow_at_least_1px',False)):
        rows=[r for r in evaluation if r.get('seed_base')==cfg['seed'] and r.get('status')=='completed'
              and (r['metrics']['source_motion_proxy_px']<1.)==low and r['arm'] in cfg['main_arms']]
        strata[label]=aggregate(rows,cfg['main_arms'])
    # Every selected source appears, including failures, not just favorable demos.
    page=['<!doctype html><meta charset="utf-8"><title>Frozen paper evidence</title>',
          '<style>body{font:16px Arial;max-width:1400px;margin:25px auto}video{width:100%}pre{white-space:pre-wrap}</style>',
          '<h1>Reserved-cohort editing evidence</h1><p>Same input, anchor and seed; full coverage below.</p>']
    groups={'main':['native','rgb_full','rae_full'],'reference_sft':['native_reference','sft','rae_full'],
            'state_feedback':['rae_base','rae_full','rgb_full']}
    for row in selected['records']:
        page.append('<h2>'+escape(row['id']+' '+row['instruction'])+'</h2>')
        receipt='anchors/'+row['id']+'/receipt.json'
        try:
            roots=reconcile_case(out,receipt,m['identity'])
            sync.append(dict(receipt=receipt,ok=all(r['ok'] for r in roots),roots=roots))
        except Exception as exc:sync.append(dict(receipt=receipt,ok=False,error=repr(exc)))
        for label,members in groups.items():
            try:
                payloads=[out.load(f'render/{arm}/seed{cfg["seed"]}/{row["id"]}/result.pt',m['identity'],True) for arm in members]
                if len({p['input_identity'] for p in payloads})!=1:raise ValueError('Unmatched report inputs')
                folder='comparisons/'+row['id']+'/'+label;receipt=folder+'/receipt.json';movie=folder+'/comparison.mp4'
                prior=recover_case(out.local,receipt,m['identity'],out.roots)
                generations=[p['generation_identity'] for p in payloads]
                if prior is None or prior['metadata'].get('generations')!=generations:
                    rect=payloads[0]['spatial']['valid_rect']
                    write_panels(out.local/movie,[crop_video(p['generated'],rect) for p in payloads],members,cfg['fps'])
                    seal(out.local,receipt,m['identity'],[movie],dict(generations=generations))
                out.publish([movie,receipt]);roots=reconcile_case(out,receipt,m['identity'])
                sync.append(dict(receipt=receipt,ok=all(r['ok'] for r in roots),roots=roots))
                page.append('<h3>'+escape(label)+'</h3><video controls preload="none" src="'+movie+'"></video>')
            except Exception as exc:
                failures.append(dict(id=row['id'],phase='comparison',comparison=label,error=repr(exc)))
                page.append('<p>'+escape(label+': unavailable '+repr(exc))+'</p>')
    human=human_packet(out,m,cfg,selected)
    bad=[r for r in sync if not r['ok']]
    if human['failures'] or any(not r['ok'] for r in human['roots']):failures.append(dict(phase='human_export',report=human))
    training=arm_store(out,'sft').json('training_summary.json')
    if not training or training.get('status')!='completed':failures.append(dict(phase='sft',status=training))
    report=dict(status='completed' if not failures and not bad else 'incomplete',identity=m['identity'],
        cohort={k:v for k,v in selected.items() if k!='records'},coverage=coverage,tables=tables,runtime=runtime,
        sft=training,sft_matching=arm_store(out,'sft').json('matching.json'),human=human,
        metric_coverage=metric_coverage,input_motion_strata=strata,
        motion_strata_caveat='Fixed 1-pixel mean source-flow threshold at <=256px; a motion proxy, not object/camera ground truth',
        automated_metrics_status='complete' if all(n==sum(v['expected'] for v in coverage.values()) for k,n in metric_coverage.items() if k!='frame_delta_ratio') else 'partial',
        evaluator_status=sorted({json.dumps(r.get('lpips',{}),sort_keys=True) for r in evaluation if r.get('status')=='completed'}),
        failures=failures,media_failures=bad,limitations=LIMITS,
        compute='sampling_seconds includes online solves, decode and intermediate publication; shared preprocessing and loading are separate')
    out.write_json('summary.json',report);out.write_json('media_sync.json',dict(checked=len(sync),records=sync,failed=len(bad)))
    page.append('<pre>'+escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(page),encoding='utf8');out.publish(['report.html'])
    if failures or bad:raise RuntimeError('Paper suite incomplete; successful cases retained, inspect summary.json')
