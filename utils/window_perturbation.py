"""Frozen decoder stress test in normalized latent coordinates."""
import time
import torch
from utils.window_training import inverse
from utils.training import append_metrics, atomic_torch_save
from utils.video_preview import save_video_preview

ALPHAS = (0., .05, .1, .25, .5, 1.)


def matched_random(error, seed):
    """Match RMS separately per batch/future slot; no global RNG mutation."""
    random = torch.randn(error.shape, generator=torch.Generator().manual_seed(seed)).to(error.device)
    dims = tuple(range(2, error.ndim))
    scale = error.square().mean(dims, keepdim=True).sqrt()
    return random * scale / random.square().mean(dims, keepdim=True).sqrt().clamp_min(1e-12)


@torch.no_grad()
def audit_sample(*, model, flow, c, y, cr, ae, raw, stats, decode, sync,
                 dtype, seeds, index, video_id, out, rank, previews, status):
    rows = []
    for seed in seeds:
        status('perturbation_sampling', clip=index, seed=seed)
        noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed+1009*index)).to(y.device)
        sync(); start = time.perf_counter()
        generated = flow.sample(model, c, noise, 64, dtype, method='euler')
        sync(); sample_seconds = time.perf_counter()-start
        error = generated-y
        random = matched_random(error, seed+1009*index+1000003)
        # Save all samples' inputs/directions, not just preview examples.
        name = f'eval_clip{index}_seed{seed}'
        atomic_torch_save(dict(video_id=video_id, seed=seed, cond_native=cr.cpu(),
            target_normalized=y.cpu(), generated_normalized=generated.cpu(),
            random_direction_normalized=random.cpu(), statistics={k:v.cpu() for k,v in stats.items()},
            alphas=ALPHAS, coordinate_system='normalized target; decoder anchor native/original'),
            str(out/'latents'/f'{name}.pt'))
        for direction, delta in [('generated', error), ('random', random)]:
            for alpha in ALPHAS:
                status('perturbation_decode', clip=index, seed=seed, direction=direction, alpha=alpha)
                z = y + alpha*delta
                sync(); start = time.perf_counter()
                rgb = decode(cr, inverse(z, stats))
                sync(); decode_seconds = time.perf_counter()-start
                difference = rgb[:, 1:]-ae[:, 1:]
                latent_mse = float((z-y).square().mean())
                rgb_mse = float(difference.square().mean())
                row = dict(split='eval', index=index, video_id=video_id, seed=seed,
                    arm=f'{direction}_a{alpha:g}', direction=direction, alpha=alpha,
                    latent_mse=latent_mse, rgb_mse_vs_ae=rgb_mse,
                    rgb_l1_vs_ae=float(difference.abs().mean()),
                    rgb_l1_vs_raw=float((rgb[:,1:]-raw[:,1:]).abs().mean()),
                    per_frame_l1_vs_ae=difference.abs().mean((0,2,3,4)).cpu().tolist(),
                    per_frame_l1_vs_raw=(rgb[:,1:]-raw[:,1:]).abs().mean((0,2,3,4)).cpu().tolist(),
                    rgb_rms_over_latent_rms=(rgb_mse/latent_mse)**.5 if latent_mse>0 else None,
                    generated_motion=float(rgb.diff(dim=1).abs().mean()),
                    sample_seconds=sample_seconds, decode_seconds=decode_seconds,
                    DI_throughput=y.shape[1]*y.shape[2]*64/sample_seconds)
                append_metrics(out/f'metrics_rank{rank:03d}.jsonl', row); rows.append(row)
                if index < previews:
                    save_video_preview(str(out/'samples'), name+'_'+row['arm'],
                        dict(ae=ae[0], perturbed=rgb[0], raw=raw[0]),
                        metadata=row, fps=8, save_mp4=True)
    return rows
