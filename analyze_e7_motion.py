#!/usr/bin/env python3
"""Diagnose z0-copying in E7 samples: per-frame motion magnitude, gen vs GT.

The generator conditions on z_0; the classic failure mode is predicting
z_t ~= z_0 (frame-0 copying) instead of dynamics. Discriminator: in the
UNNORMALIZED latent space, motion magnitude ||z_t - z_0|| per future frame,
generated vs target.

  motion_ratio ~= 1 and growing with t  -> real dynamics learned
  motion_ratio << 1 / flat in t         -> z0-copy shortcut confirmed

Supports local dirs and obs:// (moxing). Sample dirs accumulate .pt files
from many eval steps; only the LATEST step is analyzed unless --step is
given.

Cluster usage:
    python analyze_e7_motion.py \
        --samples obs://.../output/scale/dual_diffusion_absolute/samples \
        --output  obs://.../output/scale/dual_diffusion_absolute/samples/motion_report.txt
"""

from __future__ import annotations

import argparse
import io
import os
import re
import tempfile

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', required=True,
                   help='dir of step*_clip*.pt (local or obs://), or one file')
    p.add_argument('--step', type=int, default=-1,
                   help='eval step to analyze; -1 = latest found')
    p.add_argument('--output', default='',
                   help='optional report path (local or obs://)')
    return p.parse_args()


def _is_remote(path):
    return path.startswith(('obs://', 's3://'))


def list_sample_files(samples):
    if _is_remote(samples):
        import moxing as mox
        names = mox.file.list_directory(samples)
        prefix = samples.rstrip('/') + '/'
        names = [n[len(prefix):] if n.startswith(prefix) else n
                 for n in names]
        return sorted(n for n in names if n.endswith('.pt'))
    if os.path.isdir(samples):
        return sorted(n for n in os.listdir(samples) if n.endswith('.pt'))
    return [os.path.basename(samples)]


def load_pack(samples, name, tmp_dir):
    if _is_remote(samples):
        import moxing as mox
        local = os.path.join(tmp_dir, name)
        mox.file.copy(samples.rstrip('/') + '/' + name, local)
        return torch.load(local, map_location='cpu', weights_only=False)
    base = samples if os.path.isdir(samples) else os.path.dirname(samples)
    return torch.load(os.path.join(base, name), map_location='cpu',
                      weights_only=False)


def denorm(x1, stats, target_mode, z0):
    if target_mode == 'absolute':
        m = stats['abs_mean'][1:].unsqueeze(0).unsqueeze(2)
        s = stats['abs_std'][1:].unsqueeze(0).unsqueeze(2)
        return x1 * s + m
    m = stats['res_mean'].unsqueeze(0).unsqueeze(2)
    s = stats['res_std'].unsqueeze(0).unsqueeze(2)
    return z0 + (x1 * s + m)


def main():
    args = parse_args()
    names = list_sample_files(args.samples)
    steps = sorted({int(m.group(1)) for n in names
                    if (m := re.match(r'step(\d+)_clip\d+\.pt$', n))})
    if not steps:
        raise SystemExit(f'no step*_clip*.pt under {args.samples}')
    step = args.step if args.step >= 0 else steps[-1]
    picked = [n for n in names
              if re.match(rf'step0*{step}_clip\d+\.pt$', n)]
    if not picked:
        raise SystemExit(f'step {step} not found; available: {steps}')

    gen_motion, tgt_motion, cos_rows = [], [], []
    mode = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in picked:
            pack = load_pack(args.samples, name, tmp_dir)
            z0 = pack['z0_unnorm'].float()                # [B,1,N,C]
            mode = pack['target_mode']
            z_gen = denorm(pack['sampled_x1'].float(), pack['stats'],
                           mode, z0)
            z_tgt = denorm(pack['target_x1'].float(), pack['stats'],
                           mode, z0)
            d_gen = z_gen - z0                            # [B,7,N,C]
            d_tgt = z_tgt - z0
            gen_motion.append(d_gen.flatten(2).norm(dim=2).mean(0))
            tgt_motion.append(d_tgt.flatten(2).norm(dim=2).mean(0))
            cos_rows.append(torch.nn.functional.cosine_similarity(
                d_gen.flatten(2), d_tgt.flatten(2), dim=2).mean(0))

    gen_m = torch.stack(gen_motion).mean(0)
    tgt_m = torch.stack(tgt_motion).mean(0)
    cos_m = torch.stack(cos_rows).mean(0)
    ratio = gen_m / tgt_m.clamp(min=1e-8)

    buf = io.StringIO()
    buf.write(f'samples: {args.samples}\n')
    buf.write(f'step {step}, {len(picked)} clips, target_mode={mode}\n\n')
    buf.write(f'{"frame":>5} {"||gen-z0||":>12} {"||gt-z0||":>12} '
              f'{"ratio":>7} {"cos(dir)":>9}\n')
    for t in range(gen_m.numel()):
        buf.write(f'{t + 1:>5} {gen_m[t]:>12.3f} {tgt_m[t]:>12.3f} '
                  f'{ratio[t]:>7.3f} {cos_m[t]:>9.3f}\n')
    buf.write(f'\nmean motion ratio: {ratio.mean():.3f}  '
              f'(~1 = matches GT motion; <<1 = z0-copying)\n')
    buf.write(f'GT motion growth f1->f7: {tgt_m[0]:.3f} -> {tgt_m[-1]:.3f}; '
              f'gen: {gen_m[0]:.3f} -> {gen_m[-1]:.3f} '
              f'(healthy: gen grows with t like GT)\n')
    report = buf.getvalue()
    print(report)

    if args.output:
        if _is_remote(args.output):
            import moxing as mox
            with tempfile.NamedTemporaryFile(
                    'w', suffix='.txt', delete=False) as handle:
                handle.write(report)
                local = handle.name
            mox.file.copy(local, args.output)
            os.unlink(local)
        else:
            with open(args.output, 'w') as handle:
                handle.write(report)
        print(f'report -> {args.output}')


if __name__ == '__main__':
    main()
