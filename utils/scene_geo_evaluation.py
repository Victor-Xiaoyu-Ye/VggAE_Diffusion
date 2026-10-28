"""Paired same-noise condition interventions for the explicitly oracle arm."""
from __future__ import annotations

import torch

from utils.scene_prefix_eval import sample_case


@torch.no_grad()
def condition_diagnostics(model, rae, sample, samples, mean, std, device,
                          runtime, cfg, seed, correct_frames, correct_z,
                          store, prefix, preview, label):
    if runtime.arm != 'oracle_geo':
        return dict(status='not_applicable', reason='prefix-only baseline'), []
    channels = sample['target'].shape[-1]
    half = channels // 2
    history = cfg['history']
    candidates = [s for s in samples if s['id'] != sample['id']
                  and s['target'].shape == sample['target'].shape]
    if not candidates:
        return dict(status='failed', error='No distinct same-codec geometry donor'), []
    donor = sorted(candidates, key=lambda s: (s['id'], str(s.get('frame_ids'))))[0]
    donor_geo = donor['target'][history:, :, :half].detach().clone()
    width = correct_frames.shape[2]//4
    reference = correct_frames[history:, :, 2*width:3*width]
    files, rows = [], {}
    for variant in ('zero', 'shuffle'):
        altered = dict(sample, geo_condition_variant=variant, donor_geo=donor_geo,
                       donor_case_id=donor['id'])
        try:
            row, frames, z = sample_case(model, rae, altered, mean, std, device,
                history=history, steps=cfg['sample_steps'], seed=seed, net_factory=runtime.bind)
            runtime.annotate(row, altered)
            row.update(rgb_change_from_correct_l1=float((frames[history:, :, 2*width:3*width]-reference).abs().mean()),
                generated_geo_change_from_correct_l1=float((z[..., :half]-correct_z[..., :half]).abs().mean()),
                generated_texture_change_from_correct_l1=float((z[..., half:]-correct_z[..., half:]).abs().mean()),
                diagnostic='same-noise oracle condition intervention; changes alone are not improved quality')
            name = prefix+f'_GEO_{variant.upper()}'
            display = label(frames, ['RAW', 'AE', 'ORACLE GEO '+variant.upper(), 'COPY'], history)
            media = preview(store, name+'.mp4', display)
            row['preview_status'] = 'completed' if name+'.mp4' in media else 'failed'
            store.save(name+'.pt', dict(latent=z, frame_ids=sample['frame_ids'], variant=variant,
                seed=seed, donor_case_id=donor['id']), runtime.run_identity)
            store.write_json(name+'.json', row)
            files += media+[name+'.pt', name+'.pt.json', name+'.json']
            rows[variant] = row
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            rows[variant] = dict(error=repr(exc))
    status = 'completed' if all('error' not in r and r.get('preview_status') == 'completed'
                                for r in rows.values()) else 'failed'
    return dict(status=status, donor_case_id=donor['id'], variants=rows,
                note='Correct/zero/wrong geometry use identical initial noise. No GT geometry is substituted into decoded output.'), files
