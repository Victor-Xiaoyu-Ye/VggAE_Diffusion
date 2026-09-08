#!/usr/bin/env python3
"""Read-only real-codec n1 oracle and historical deterministic sampler validation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from models.single_target_generator import SingleTargetGenerator
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, resolve_dtype, manual_seed_all)
from utils.file_signature import sampled_file_signature
from utils.flow_run_status import atomic_json
from utils.latent_generation_metrics import (fit_statistics, to_device,
                                            normalize, inverse, complete_prefix)
from utils.sampler_contract import ConstantClean, DeterministicClean, check_endpoint
from utils.video_preview import save_video_preview


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('eval_csv', 'video_root', 'encoder_ckpt', 'r7_ckpt',
                 'deterministic_ckpt', 'output_dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--dtype', choices=('fp32', 'bf16', 'fp16'), default='bf16')
    parser.add_argument('--sample_steps', default='1,30,60')
    parser.add_argument('--sample_seeds', default='42,43,44,45')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def run(args):
    steps = [int(x) for x in args.sample_steps.split(',')]
    seeds = [int(x) for x in args.sample_seeds.split(',')]
    if not steps or min(steps) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError('positive steps and distinct seeds required')
    backend = get_device_name()
    if backend == 'cpu':
        raise RuntimeError('Real codec validation requires GPU/NPU; use the CPU unit tests separately')
    # Keep --help independent of video/encoder optional dependencies.
    from train_single_target_probe import FrozenR7, build_loader, psnr
    from train_r7_flow_probe import pack_samples, stack
    configure_backend_compatibility(backend)
    manual_seed_all(42)
    device, dtype = get_device(0), resolve_dtype(args.dtype)
    frozen = FrozenR7(args.encoder_ckpt, args.r7_ckpt, device, dtype)
    items = pack_samples(build_loader(args.eval_csv, args.video_root,
        frozen.config, 1, 1, args.num_workers, False), frozen, 1)
    if len(items) != 1:
        raise ValueError('n1 requires exactly one materialized clip')
    del frozen.encoder, frozen.compressor, frozen.tex_encoder
    from utils.device import empty_cache
    empty_cache()
    c, y = stack(items, device)
    cs, ys = [to_device(fit_statistics(x), device) for x in (c, y)]
    cn, yn = normalize(c, cs), normalize(y, ys)
    payload = torch.load(args.deterministic_ckpt, map_location='cpu', weights_only=False)
    if (payload.get('schema') != 'r7-single-target-probe-v1'
            or payload.get('mode') != 'deterministic'
            or payload.get('representation') != frozen.representation_contract
            or payload['args'].get('target_index') != 1):
        raise ValueError('deterministic checkpoint schema/representation/target mismatch')
    config = payload['args']
    predictor = SingleTargetGenerator(latent_dim=frozen.config.latent_dim,
        num_tokens=frozen.config.latent_grid**2, hidden_dim=config['hidden_dim'],
        depth=config['depth'], max_target_index=frozen.config.latent_seq_len-1).to(device).eval()
    predictor.load_state_dict(payload['model'], strict=True)
    det_raw = predictor(c[:, 0], 1)[:, None]
    det_normalized = normalize(det_raw, ys)
    del payload

    def decode(z):
        return frozen.decode_full(complete_prefix(c, z, frozen.config.seq_len))[:, 1:2]

    raw = items[0]['raw_target'][None].to(device).float().div(255).permute(0, 1, 3, 4, 2)
    ae, direct = decode(y), decode(det_raw)
    roundtrip_error = float((inverse(yn, ys)-y).abs().max())
    anchor_error = float((inverse(cn, cs)-c).abs().max())
    roundtrip_rgb_error = float((decode(inverse(yn, ys))-ae).abs().max())
    report = dict(schema='r7-sampler-contract-v1', passed=False,
        generation_quality_gate=False, video_id=items[0]['video_id'],
        window_index=items[0]['window_index'], dtype=args.dtype,
        materialized_sha256=hashlib.sha256(
            items[0]['full_latent'].contiguous().numpy().tobytes()
            + items[0]['raw_target'].contiguous().numpy().tobytes()).hexdigest(),
        r7_signature=sampled_file_signature(args.r7_ckpt),
        encoder_signature=sampled_file_signature(args.encoder_ckpt),
        deterministic_signature=sampled_file_signature(args.deterministic_ckpt),
        deterministic_identity_note='Legacy checkpoint lacks materialized sample hash; current clip ID is recorded, historical identity is not certified.',
        tolerances=dict(normalized_max_abs=1e-4, rgb_max_abs=2e-3),
        target_roundtrip_max_abs=roundtrip_error,
        anchor_roundtrip_max_abs=anchor_error,
        roundtrip_rgb_max_abs=roundtrip_rgb_error,
        ae_psnr_raw=psnr(ae, raw), deterministic_psnr_raw=psnr(direct, raw),
        deterministic_psnr_ae=psnr(direct, ae), rows=[])
    previews = {'raw': raw[0], 'ae': ae[0], 'det_direct': direct[0]}
    for label, model, target in (
        ('oracle', ConstantClean(yn), yn),
        ('deterministic', DeterministicClean(predictor, cs, ys), det_normalized)):
        for alpha in (1., 3.):
            rows, rgb = check_endpoint(model, cn, target, ys, decode, seeds, steps, dtype, alpha)
            report['rows'].extend(dict(arm=label, **row) for row in rows)
            previews[f'{label}_grid{alpha:g}'] = rgb[0]
    report['passed'] = (all(row['passed'] for row in report['rows'])
                        and roundtrip_error <= 1e-4 and anchor_error <= 1e-4
                        and roundtrip_rgb_error <= 2e-3)
    save_video_preview(str(Path(args.output_dir)/'samples'), 'sampler_contract',
                       previews, save_mp4=False)
    return report


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = output/'contract_status.json'
    # Refuse overwrite even for direct invocation; failed jobs need a fresh namespace.
    with status.open('x', encoding='utf-8') as handle:
        json.dump(dict(schema='r7-sampler-contract-v1', passed=False, status='running'), handle)
    try:
        report = run(args)
        report['status'] = 'completed'
        atomic_json(status, report)
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise SystemExit('Sampler endpoint contract failed; do not start training')
    except Exception as exc:
        atomic_json(status, dict(schema='r7-sampler-contract-v1', passed=False,
                                 status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
