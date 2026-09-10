"""GT-assisted frozen-decoder oracle, never a deployable generator."""
import time
import torch
from utils.window_training import normalize, inverse
from utils.training import append_metrics, atomic_torch_save
from utils.video_preview import save_video_preview

PROJECTORS = ('frequency_low', 'frequency_mid', 'frequency_high', 'pca8', 'pca32')
ARMS = ('generated', 'target') + tuple(a for p in PROJECTORS for a in (p, p+'_matched_residual'))


def fit_basis(samples, stats):
    """CPU float64 covariance, training samples only, no heldout fitting."""
    channels = stats['mean'].shape[-1]
    total = torch.zeros(channels, dtype=torch.float64)
    outer = torch.zeros(channels, channels, dtype=torch.float64)
    count = 0
    for sample in samples:
        x = normalize(sample['target'][None], stats).reshape(-1, channels).double()
        total += x.sum(0); outer += x.T @ x; count += len(x)
    if count < 2:
        raise ValueError('insufficient PCA samples')
    cov = (outer-total[:, None]*total[None, :]/count)/(count-1)
    values, basis = torch.linalg.eigh(cov)
    return basis.flip(1).float(), values.flip(0).float()


def removed_component(error, arm, basis):
    # FFT/eigh support varies on Ascend; the small diagnostic runs on CPU.
    x = error.detach().cpu().float()
    if arm.startswith('pca'):
        v = basis[:, :int(arm[3:])].cpu().float()
        return ((x @ v) @ v.T).to(error.device)
    grid = int(x.shape[2]**.5)
    if grid*grid != x.shape[2]:
        raise ValueError('expected square latent grid')
    f = torch.fft.fftfreq(grid)
    radius = (f[:, None].square()+f[None, :].square()).sqrt()
    masks = {'frequency_low': radius <= .15,
             'frequency_mid': (radius > .15) & (radius <= .3),
             'frequency_high': radius > .3}
    mask = masks[arm][None, None, :, :, None]
    z = x.reshape(x.shape[0], x.shape[1], grid, grid, x.shape[-1])
    result = torch.fft.ifft2(torch.fft.fft2(z, dim=(2, 3), norm='ortho')*mask,
                             dim=(2, 3), norm='ortho').real.reshape_as(x)
    return result.to(error.device)


def repairs(y, generated, basis):
    error = generated-y
    yield 'generated', generated
    yield 'target', y
    power = error.square().mean((2, 3), keepdim=True)
    for arm in PROJECTORS:
        repaired = generated-removed_component(error, arm, basis)
        residual_power = (repaired-y).square().mean((2, 3), keepdim=True)
        # Same remaining MSE per future slot, but shrink every original direction.
        scale = (residual_power/power.clamp_min(1e-20)).sqrt()
        yield arm, repaired
        yield arm+'_matched_residual', y+scale*error


@torch.no_grad()
def audit_sample(*, model, flow, c, y, cr, ae, raw, stats, decode, sync,
                 dtype, seeds, index, video_id, out, rank, previews, status, basis):
    rows = []
    for seed in seeds:
        status('subspace_sampling', clip=index, seed=seed)
        noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed+1009*index)).to(y.device)
        sync(); start = time.perf_counter()
        generated = flow.sample(model, c, noise, 64, dtype, method='euler')
        sync(); seconds = time.perf_counter()-start
        name = f'eval_clip{index}_seed{seed}'
        atomic_torch_save(dict(video_id=video_id, seed=seed, cond_native=cr.cpu(),
            target_normalized=y.cpu(), generated_normalized=generated.cpu(),
            statistics={k:v.cpu() for k,v in stats.items()}, oracle=True), str(out/'latents'/f'{name}.pt'))
        for arm, z in repairs(y, generated, basis):
            status('subspace_decode', clip=index, seed=seed, arm=arm)
            sync(); start = time.perf_counter()
            rgb = decode(cr, inverse(z, stats))
            sync(); decode_seconds = time.perf_counter()-start
            difference = rgb[:, 1:]-ae[:, 1:]
            row = dict(split='eval', index=index, video_id=video_id, seed=seed,
                arm=arm, oracle=True, latent_mse=float((z-y).square().mean()),
                rgb_l1_vs_ae=float(difference.abs().mean()),
                rgb_l1_vs_raw=float((rgb[:, 1:]-raw[:, 1:]).abs().mean()),
                per_frame_l1_vs_ae=difference.abs().mean((0, 2, 3, 4)).cpu().tolist(),
                generated_motion=float(rgb.diff(dim=1).abs().mean()),
                sample_seconds=seconds, decode_seconds=decode_seconds,
                DI_throughput=y.shape[1]*y.shape[2]*64/seconds)
            append_metrics(out/f'metrics_rank{rank:03d}.jsonl', row); rows.append(row)
            if index < previews:
                save_video_preview(str(out/'samples'), name+'_'+arm,
                    dict(ae=ae[0], oracle=rgb[0], raw=raw[0]), metadata=row, fps=8, save_mp4=True)
    return rows
