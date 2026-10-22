"""Offline official VGGT teacher for track-mask diagnostics, never generation inputs.

This is deliberately separate from the production causal StreamVGGT encoder.
No weights are downloaded, and every required camera/depth/tracking weight must
exist and match before inference. All returned coordinates refer to the supplied
RGB tensor; this adapter performs no implicit resize or crop.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch


VENDORED_ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'vggt_trackmask'
UPSTREAM_COMMIT = 'a288dd0f14786c93483e45524328726ab7b1b4ce'
REQUIRED_GROUPS = ('aggregator', 'camera_head', 'depth_head', 'track_head')


def import_official_vggt(code_root=None):
    """Import only from the explicit official checkout, rejecting module collisions."""
    root = Path(code_root or VENDORED_ROOT).resolve()
    required = ('models/vggt.py', 'heads/track_head.py', 'heads/camera_head.py',
                'heads/dpt_head.py', 'utils/pose_enc.py')
    for rel in required:
        if not (root / 'vggt' / rel).is_file():
            raise FileNotFoundError(f'Full official VGGT code required: {root / "vggt" / rel}')
    for name, module in list(sys.modules.items()):
        if name == 'vggt' or name.startswith('vggt.'):
            paths = list(getattr(module, '__path__', []))
            if getattr(module, '__file__', None):
                paths.append(module.__file__)
            if any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise RuntimeError('Another vggt package is already imported; use a fresh process')
    sys.path.insert(0, str(root))
    try:
        model_module = importlib.import_module('vggt.models.vggt')
        pose_module = importlib.import_module('vggt.utils.pose_enc')
    except ImportError as exc:
        raise RuntimeError('Official VGGT requires torch, numpy, einops and huggingface_hub; '
                           'install missing dependencies without replacing cluster torch') from exc
    finally:
        sys.path.remove(str(root))
    if not Path(model_module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Imported VGGT does not belong to requested checkout')
    digest = hashlib.sha256()
    for path in sorted((root / 'vggt').rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    provenance = {'code_root': str(root), 'code_sha256': digest.hexdigest(),
                  'upstream_commit': UPSTREAM_COMMIT if root == VENDORED_ROOT.resolve() else None}
    return model_module.VGGT, pose_module.pose_encoding_to_extri_intri, provenance


def checkpoint_state(payload):
    """Accept plain official weights or a conventional state_dict/model wrapper."""
    if not isinstance(payload, Mapping):
        raise TypeError('Checkpoint must contain a tensor state dictionary')
    for name in ('state_dict', 'model'):
        if name in payload and isinstance(payload[name], Mapping):
            payload = payload[name]
            break
    if not payload or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
        raise TypeError('Checkpoint is not a pure tensor state dictionary')
    if all(k.startswith('module.') for k in payload):
        payload = {k[len('module.'):]: v for k, v in payload.items()}
    return payload


def validate_checkpoint_coverage(model, state):
    """Require the entire active backbone and heads, not a permissive strict=False load."""
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    mismatched = sorted(k for k in expected.keys() & state.keys()
                        if tuple(expected[k].shape) != tuple(state[k].shape))
    # The unused official point head may exist in the full published checkpoint.
    unexpected = sorted(k for k in set(state) - set(expected) if not k.startswith('point_head.'))
    groups = {}
    for prefix in REQUIRED_GROUPS:
        names = [k for k in expected if k.startswith(prefix + '.')]
        matched = [k for k in names if k in state and k not in mismatched]
        groups[prefix] = {'required_keys': len(names), 'matched_keys': len(matched),
                          'required_numel': sum(expected[k].numel() for k in names)}
        if not names:
            missing.append(prefix + '.* (model has no required module)')
    report = {'groups': groups, 'missing': missing, 'shape_mismatches': mismatched,
              'unexpected': unexpected,
              'ignored_point_head_keys': sum(k.startswith('point_head.') for k in state if k not in expected)}
    if missing or mismatched or unexpected:
        raise ValueError('Incompatible full VGGT checkpoint; no randomly initialized heads are allowed: '
                         + json.dumps(report, ensure_ascii=False))
    return report


def _device(device):
    value = str(device)
    if value.startswith('npu'):
        try:
            importlib.import_module('torch_npu')
        except ImportError as exc:
            raise RuntimeError('NPU requested but torch_npu is unavailable') from exc
        if not torch.npu.is_available():
            raise RuntimeError('NPU requested but no NPU is available')
    result = torch.device(value)
    if result.type not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Teacher supports explicitly selected cpu/cuda/npu devices')
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA device is available')
    if result.type == 'cuda':
        torch.cuda.set_device(result)
    elif result.type == 'npu':
        torch.npu.set_device(result)
    return result


def query_grid(height, width, stride=21, margin=7):
    if stride < 1 or margin < 0 or 2 * margin >= min(height, width):
        raise ValueError('Invalid query grid stride or margin')
    y = torch.arange(margin, height - margin, stride, dtype=torch.float32)
    x = torch.arange(margin, width - margin, stride, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def validate_predictions(outputs, frames, queries):
    t, _, h, w = frames.shape
    n = len(queries)
    shapes = {'depth': (t, h, w), 'depth_conf': (t, h, w),
              'extrinsics': (t, 3, 4), 'intrinsics': (t, 3, 3),
              'tracks': (t, n, 2), 'visibility': (t, n), 'track_confidence': (t, n)}
    for name, shape in shapes.items():
        tensor = outputs[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f'Invalid teacher {name} shape {tuple(tensor.shape)} != {shape}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite teacher {name}; mark this case failed, do not fabricate a mask')
    if (outputs['depth'] <= 0).any():
        raise ValueError('Teacher depth must be positive')
    for name in ('visibility', 'track_confidence'):
        if ((outputs[name] < 0) | (outputs[name] > 1)).any():
            raise ValueError(f'Teacher {name} must contain probabilities in [0,1]')
    if (outputs['intrinsics'][:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError('Teacher focal lengths must be positive')


class VGGTTrackTeacher:
    """Full bidirectional VGGT diagnostics; NEVER a causal generation conditioner."""

    def __init__(self, checkpoint, code_root=None, device='cpu', precision='float32',
                 query_stride=21, query_margin=7, query_chunk_size=0):
        self.device = _device(device)
        if precision not in ('float32', 'bfloat16'):
            raise ValueError('precision must be float32 or bfloat16')
        if self.device.type == 'cpu' and precision != 'float32':
            raise ValueError('Use float32 for CPU teacher diagnostics')
        if query_chunk_size < 0:
            raise ValueError('query_chunk_size must be nonnegative (0 means all queries together)')
        self.precision = precision
        self.query_stride, self.query_margin = int(query_stride), int(query_margin)
        if self.query_stride < 1 or self.query_margin < 0:
            raise ValueError('query_stride must be positive and query_margin nonnegative')
        self.query_chunk_size = int(query_chunk_size)
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Explicit local full VGGT checkpoint required: {path}')
        model_class, self.pose_decoder, provenance = import_official_vggt(code_root)
        # Point-head outputs are unnecessary; published point weights are explicitly ignored.
        self.model = model_class(enable_camera=True, enable_point=False, enable_depth=True, enable_track=True)
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(path), device='cpu')
        else:
            state = checkpoint_state(torch.load(str(path), map_location='cpu', weights_only=True,
                                                mmap=os.name != 'nt'))
        coverage = validate_checkpoint_coverage(self.model, state)
        active = {k: state[k] for k in self.model.state_dict()}
        self.model.load_state_dict(active, strict=True, assign=True)
        del state, active
        self.model.eval().requires_grad_(False).float().to(self.device)
        stat = path.stat()
        sample_digest = hashlib.sha256()
        # Bounded provenance read; explicitly labelled partial rather than a full-file hash.
        with path.open('rb') as handle:
            for offset in sorted(set((0, max(0, stat.st_size // 2 - 524288), max(0, stat.st_size - 1048576)))):
                handle.seek(offset)
                sample_digest.update(str(offset).encode() + handle.read(1048576))
        self.metadata = {**provenance, 'checkpoint': str(path), 'checkpoint_size': stat.st_size,
                         'checkpoint_mtime_ns': stat.st_mtime_ns,
                         'checkpoint_sample_sha256': sample_digest.hexdigest(),
                         'checkpoint_sample_scheme': 'offset-prefixed first/middle/last 1MiB',
                         'coverage': coverage, 'device': str(self.device), 'precision': precision,
                         'heads_precision': 'float32', 'query_stride': self.query_stride,
                         'query_margin': self.query_margin, 'query_chunk_size': self.query_chunk_size,
                         'query_chunk_changes_tracker_context': bool(self.query_chunk_size),
                         'role': 'offline_bidirectional_pseudo_label_teacher', 'implicit_resize': False}

    def _autocast(self):
        if self.precision == 'float32':
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    @torch.inference_mode()
    def infer(self, frames, reference_frame=0):
        frames = torch.as_tensor(frames)
        if frames.ndim != 4 or frames.shape[1] != 3 or len(frames) < 2:
            raise ValueError('Expected at least two RGB frames [T,3,H,W]')
        if not frames.is_floating_point() or not torch.isfinite(frames).all():
            raise ValueError('RGB frames must be finite floating point in [0,1]')
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError('RGB frames must be in [0,1]')
        t, _, h, w = frames.shape
        if h % 14 or w % 14:
            raise ValueError('Input height and width must be multiples of14; resize explicitly upstream')
        reference_frame = int(reference_frame)
        if not 0 <= reference_frame < t:
            raise ValueError('reference_frame is outside the supplied window')
        # Official VGGT uses the first frame as tracking query/camera gauge reference.
        order = [reference_frame] + [i for i in range(t) if i != reference_frame]
        inverse = torch.argsort(torch.tensor(order))
        images = frames[order].to(device=self.device, dtype=torch.float32).unsqueeze(0)
        queries = query_grid(h, w, self.query_stride, self.query_margin).to(self.device)
        with self._autocast():
            tokens, patch_start = self.model.aggregator(images)
        # Upstream model.forward disables CUDA autocast only. Explicit subhead calls
        # give CPU/CUDA/NPU the same FP32 geometry and correlation arithmetic.
        tokens = [x.float() if torch.is_tensor(x) else x for x in tokens]
        with torch.autocast(device_type=self.device.type, enabled=False):
            pose = self.model.camera_head(tokens)[-1].float()
            depth, depth_conf = self.model.depth_head(tokens, images=images, patch_start_idx=patch_start)
            extrinsics, intrinsics = self.pose_decoder(pose, (h, w))
            chunks = queries.split(self.query_chunk_size) if self.query_chunk_size else (queries,)
            tracks, visibility, confidence = [], [], []
            for points in chunks:
                track_list, vis, conf = self.model.track_head(
                    tokens, images=images, patch_start_idx=patch_start, query_points=points[None])
                tracks.append(track_list[-1]); visibility.append(vis); confidence.append(conf)
            tracks = torch.cat(tracks, dim=2)
            visibility = torch.cat(visibility, dim=2)
            confidence = torch.cat(confidence, dim=2)
        def chronological(value):
            return value[0].detach().float().cpu()[inverse].contiguous()
        result = {'depth': chronological(depth)[..., 0], 'depth_conf': chronological(depth_conf),
                  'extrinsics': chronological(extrinsics), 'intrinsics': chronological(intrinsics),
                  'tracks': chronological(tracks), 'visibility': chronological(visibility),
                  'track_confidence': chronological(confidence)}
        validate_predictions(result, frames, queries)
        result.update(query_points=queries.detach().float().cpu(), reference_frame=reference_frame,
                      image_size_hw=[h, w], teacher=dict(self.metadata),
                      teacher_context_frames=t, camera_convention='OpenCV_world_to_camera')
        return result
