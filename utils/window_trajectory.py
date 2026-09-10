"""Observe the production sampler; GT probes never affect its trajectory."""
import time
import torch
from utils.window_training import collate, validate_batch, normalize, inverse
from utils.training import atomic_torch_save, append_metrics
from utils.flow_run_status import atomic_json

NODES = (0, 8, 16, 32, 40, 48, 56, 60, 63)
ARMS = tuple(f'{kind}_{i:02d}' for i in NODES for kind in ('state', 'x0', 'gt_probe')) + ('final',)


@torch.no_grad()
def capture(model, flow, c, y, noise, dtype):
    snapshots = {}
    def observe(i, u, x, clean):
        if i in NODES:
            snapshots[i] = dict(u=u, state=x.detach().cpu().clone(), x0=clean.detach().cpu().clone())
    # The exact production Euler implementation, with a read-only observer.
    final = flow.sample(model, c, noise, 64, dtype, method='euler', observer=observe)
    for i, item in snapshots.items():
        u = torch.full((len(y),), item['u'], device=y.device)
        x = (1-item['u'])*y+item['u']*noise
        with torch.autocast(y.device.type, dtype=dtype, enabled=dtype != torch.float32):
            prediction = model(x, u, c, None, None)
        item['gt_probe'] = flow.clean(prediction, x, u).cpu()
    return snapshots, final


@torch.no_grad()
def run_audit(*, model, flow, online_state, data, memory_count, model_args, stats,
              decode, dtype, device, rank, world, ddp, out, seeds, previews, status):
    from torch import distributed as dist
    ema_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    st = {k:{n:stats[k][n].to(device).float() for n in ('mean','std')} for k in ('cond','target')}
    def sync():
        if device.type != 'cpu': getattr(torch, device.type).synchronize()
    rows, preview_jobs = [], []
    for label, state in [('online', online_state), ('ema', ema_state)]:
        model.load_state_dict(state, strict=True)
        for split_no, (split, samples) in enumerate(data.items()):
            for index, sample in enumerate(samples):
                if (split_no*len(samples)+index) % world != rank: continue
                batch = collate([sample])
                cr, yr = validate_batch(batch, model_args['future'], model_args['grid'], model_args['channels'])
                cr, yr = cr.to(device).float(), yr.to(device).float()
                c, y = normalize(cr, st['cond']), normalize(yr, st['target'])
                ae = decode(cr, yr)
                raw = batch.get('rgb')
                raw = raw.to(device).float().div(255).permute(0,1,3,4,2) if raw is not None else None
                # Match stage37's global evaluation index, including heldout offset64.
                noise_index = index if split == 'memorization' else memory_count+index
                for seed in seeds:
                    name = f'{split}_{index:02d}_{label}_seed{seed}'
                    status('trajectory_sampling', split=split, index=index, weights=label, seed=seed)
                    noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed+1009*noise_index)).to(device)
                    sync(); start = time.perf_counter()
                    snapshots, final = capture(model, flow, c, y, noise, dtype)
                    sync(); elapsed = time.perf_counter()-start
                    artifact = out/'latents'/f'{name}.pt'
                    atomic_torch_save(dict(snapshots=snapshots, final=final.cpu(), target=y.cpu(),
                        cond_native=cr.cpu(), statistics={k:v.cpu() for k,v in st['target'].items()},
                        video_id=sample['video_id'], weights=label, seed=seed, split=split), str(artifact))
                    states = [(f'{kind}_{i:02d}', item[kind], item['u'])
                              for i,item in snapshots.items() for kind in ('state','x0','gt_probe')]
                    states.append(('final', final.cpu(), 0.))
                    for arm, zcpu, u in states:
                        status('trajectory_metrics', split=split, index=index, weights=label, seed=seed, arm=arm)
                        z = zcpu.to(device)
                        sync(); start = time.perf_counter()
                        rgb = decode(cr, inverse(z, st['target']))
                        sync(); decode_seconds = time.perf_counter()-start
                        row = dict(split=split, index=index, video_id=sample['video_id'], weights=label,
                            seed=seed, arm=arm, u=u, latent_mse=float((z-y).square().mean()),
                            rgb_l1_vs_ae=float((rgb[:,1:]-ae[:,1:]).abs().mean()),
                            per_frame_l1_vs_ae=(rgb[:,1:]-ae[:,1:]).abs().mean((0,2,3,4)).cpu().tolist(),
                            generated_motion=float(rgb.diff(dim=1).abs().mean()),
                            capture_seconds=elapsed, decode_seconds=decode_seconds,
                            diagnostic_DI_throughput=y.shape[1]*y.shape[2]*(64+len(NODES))/elapsed)
                        if raw is not None: row['rgb_l1_vs_raw']=float((rgb[:,1:]-raw[:,1:]).abs().mean())
                        append_metrics(out/f'metrics_rank{rank:03d}.jsonl',row); rows.append(row)
                    if index < previews: preview_jobs.append((artifact, sample))
    if ddp:
        gathered=[None]*world; dist.all_gather_object(gathered,rows)
        rows=[r for group in gathered for r in group]
    if rank == 0:
        expected=sum(len(v) for v in data.values())*2*len(seeds)*len(ARMS)
        keys={(r['split'],r['index'],r['weights'],r['seed'],r['arm']) for r in rows}
        if len(rows)!=expected or len(keys)!=expected: raise ValueError('incomplete trajectory matrix')
        groups={}
        for split in data:
            for label in ('online','ema'):
                for arm in ARMS:
                    part=[r for r in rows if (r['split'],r['weights'],r['arm'])==(split,label,arm)]
                    groups[f'{split}/{label}/{arm}']={k:sum(r[k] for r in part)/len(part)
                        for k in ('latent_mse','rgb_l1_vs_ae','generated_motion')}
        for row in rows: append_metrics(out/'metrics.jsonl',row)
        atomic_json(out/'summary.json',dict(complete=True, rows=len(rows), groups=groups,
            previews_complete=False, note='GT probes are oracle diagnostics. State decodes include noise. DI includes CPU snapshots and9 additional probe calls; not production sampler throughput.'))
    if ddp: dist.barrier()
    # Only after the entire metric matrix is durable: small PNG-only previews.
    from utils.video_preview import save_video_preview
    for artifact, sample in preview_jobs:
        status('trajectory_preview', artifact=artifact.name)
        saved=torch.load(artifact,map_location='cpu',weights_only=False)
        cr=saved['cond_native'].to(device)
        videos={'ae':decode(cr,inverse(saved['target'].to(device),st['target']))[0]}
        for name,z in [('one_call',saved['snapshots'][0]['x0']),('x0_u050',saved['snapshots'][32]['x0']),
                       ('x0_u025',saved['snapshots'][48]['x0']),('final',saved['final'])]:
            videos[name]=decode(cr,inverse(z.to(device),st['target']))[0]
        save_video_preview(str(out/'samples'),artifact.stem,videos,save_frames=False,save_mp4=False)
    if ddp: dist.barrier()
    if rank==0: atomic_json(out/'previews_complete.json',dict(complete=True))
