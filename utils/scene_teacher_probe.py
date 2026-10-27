"""Temporary, bounded activation probes for an existing full-head teacher.

This module does not construct models, change precision, edit weights, relax
filters, or rerun a failing inference. Callers own device/checkpoint selection.
All statistics are computed on bounded CPU samples, not FP64 NPU tensors.
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch

from scripts.audit_scene_control_teacher import inspect_context


PROBE_MODULES = (
    'track_head.feature_extractor',
    'track_head.tracker.fmap_norm',
    'track_head.tracker.ffeat_norm',
    'track_head.tracker.ffeat_updater',
    'track_head.tracker.vis_predictor',
    'track_head.tracker.conf_predictor',
)
PARAMETER_MODULES = (
    'track_head.feature_extractor.norm',
    'track_head.tracker.fmap_norm',
    'track_head.tracker.ffeat_norm',
    'track_head.tracker.vis_predictor',
    'track_head.tracker.conf_predictor',
)


def _sample_cpu(tensor, limit):
    """Select uniformly in logical C-order without flattening a huge strided view."""
    tensor = tensor.detach()
    count = tensor.numel()
    if count <= limit:
        return tensor.reshape(-1).float().cpu().numpy().astype(np.float64)
    flat_indices = torch.arange(limit, dtype=torch.int64, device='cpu') * (count - 1) // (limit - 1)
    indices = []
    for size in reversed(tensor.shape):
        indices.append((flat_indices % size).to(tensor.device))
        flat_indices = flat_indices // size
    selected = tensor[tuple(reversed(indices))]
    return selected.float().cpu().numpy().astype(np.float64)


def tensor_summary(tensor, max_samples=65536, *, logits=False):
    """JSON-safe bounded statistics; finite counts refer to the recorded sample."""
    if max_samples < 2:
        raise ValueError('max_samples must be at least two')
    values = _sample_cpu(tensor, max_samples)
    finite = np.isfinite(values)
    selected = values[finite]
    result = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
        numel=tensor.numel(), sample_count=int(values.size),
        exact=tensor.numel() <= max_samples,
        sampling='all_values' if tensor.numel() <= max_samples else 'uniform_logical_flat_indices',
        finite_sample_count=int(finite.sum()), nonfinite_sample_count=int((~finite).sum()),
        minimum=float(selected.min()) if selected.size else None,
        maximum=float(selected.max()) if selected.size else None,
        mean=float(selected.mean()) if selected.size else None,
        std=float(selected.std()) if selected.size else None,
        rms=float(np.sqrt(np.mean(selected * selected))) if selected.size else None,
        sample_l2=float(np.linalg.norm(selected)) if selected.size else None)
    if logits:
        # Stable sigmoid of the sampled pre-activation, not another model call.
        probability = np.empty_like(selected)
        positive = selected >= 0
        probability[positive] = 1 / (1 + np.exp(-selected[positive]))
        exp = np.exp(selected[~positive])
        probability[~positive] = exp / (1 + exp)
        result['sigmoid_of_sampled_logits'] = dict(
            minimum=float(probability.min()) if probability.size else None,
            median=float(np.median(probability)) if probability.size else None,
            maximum=float(probability.max()) if probability.size else None)
    return result


def _tree_tensors(value, prefix='value'):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tree_tensors(item, f'{prefix}[{index}]')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tree_tensors(item, f'{prefix}.{key}')


def probe_teacher(teacher, frames, case_id, kind, *, max_samples=65536):
    """Return ``{'report': small_json_dict, 'outputs': teacher_output_or_None}``.

    ``kind`` labels caller-provided real/repeated-static/etc. input; no image is
    transformed here. ``teacher.precision`` controls backbone autocast exactly as
    in the existing adapter. Inference/diagnostic failures are reported so the
    caller can continue other independent probes. Every installed hook is
    removed in ``finally``. No zero-support result is represented as valid
    control evidence, and this function is not a gate on normal training.
    """
    if max_samples < 2:
        raise ValueError('max_samples must be at least two')
    report = dict(schema='scene-teacher-forward-probe-v1', case_id=str(case_id), kind=str(kind),
        status='starting', teacher=dict(getattr(teacher, 'metadata', {})),
        declared_backbone_precision=str(getattr(teacher, 'precision', 'unknown')),
        statistics_max_samples=int(max_samples), events=[], parameter_diagnostics={},
        missing_modules=[], instrumentation_errors=[],
        iteration_note='ffeat_norm/updater calls identify refinement iterations; classifier hooks capture the actual final logits only.',
        quality_note='Completed inference and nonzero scores do not establish tracking accuracy.')
    handles, stacks, counts = [], {}, {}
    tracking_pass = 0
    iteration = 0
    outputs = None
    started = time.monotonic()

    def summarize(value, path, *, logits=False):
        try:
            result = tensor_summary(value, max_samples, logits=logits)
            result['path'] = path
            return result
        except Exception as exc:
            error = dict(path=path, type=type(exc).__name__, message=str(exc))
            report['instrumentation_errors'].append(error)
            return dict(path=path, statistics_error=error)

    def summaries(value, module_name, module, *, output=False):
        # DPT receives the entire backbone layer list but uses four specific
        # layers. Probe those inputs, not every unused huge activation.
        if (not output and module_name.endswith('feature_extractor') and value
                and isinstance(value[0], (tuple, list))):
            selected = getattr(module, 'intermediate_layer_idx', [])
            tensors = [(f'input.layers[{i}]', value[0][i]) for i in selected
                       if 0 <= i < len(value[0]) and torch.is_tensor(value[0][i])]
            tensors.extend((path, tensor) for path, tensor in _tree_tensors(value[1:], 'input.other'))
        else:
            tensors = list(_tree_tensors(value, 'output' if output else 'input'))
        # Modules in this probe normally have one tensor output; record bounds
        # explicitly if a custom teacher supplies an unexpectedly large tree.
        limit = 8
        if len(tensors) > limit:
            report['instrumentation_errors'].append(dict(path=module_name,
                type='TensorTreeTruncated', message=f'{len(tensors)} tensors; first {limit} recorded'))
        classifier = module_name.endswith(('vis_predictor', 'conf_predictor'))
        return [summarize(tensor, path, logits=output and classifier)
                for path, tensor in tensors[:limit]]

    def enter_track(_module, _args):
        nonlocal tracking_pass, iteration
        tracking_pass += 1
        iteration = 0

    def pre_hook(name):
        def before(module, args):
            nonlocal iteration
            if name.endswith('ffeat_norm'):
                iteration += 1
            counts[name] = counts.get(name, 0) + 1
            event = dict(module=name, call=counts[name], tracking_pass=tracking_pass,
                refinement_iteration=iteration if name.endswith(('ffeat_norm', 'ffeat_updater')) else None,
                phase='entered', input=summaries(args, name, module))
            report['events'].append(event)
            stacks.setdefault(name, []).append(event)
        return before

    def post_hook(name):
        def after(module, _args, output):
            event = stacks[name].pop()
            event['output'] = summaries(output, name, module, output=True)
            event['phase'] = 'returned'
        return after

    try:
        if not torch.is_tensor(frames) or frames.ndim != 4 or len(frames) < 2:
            raise ValueError('Expected actual input frames [T,3,H,W], T>=2')
        report['input'] = summarize(frames, 'frames')
        report['input_frames_identical'] = all(torch.equal(frames[i], frames[0]) for i in range(1, len(frames)))
        model = teacher.model
        modules = {}
        for name in PROBE_MODULES:
            try:
                modules[name] = model.get_submodule(name)
            except (AttributeError, KeyError):
                report['missing_modules'].append(name)
        for name in PARAMETER_MODULES:
            try:
                module = model.get_submodule(name)
            except (AttributeError, KeyError):
                continue
            report['parameter_diagnostics'][name] = {
                key: summarize(parameter, name+'.'+key)
                for key, parameter in module.named_parameters()}
        if report['missing_modules']:
            raise ValueError('Cannot instrument required teacher modules: '+', '.join(report['missing_modules']))
        handles.append(model.get_submodule('track_head').register_forward_pre_hook(enter_track))
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))
        with torch.inference_mode():
            outputs = teacher.infer(frames, reference_frame=0)
        report['tracking_passes'] = tracking_pass
        report['hook_calls'] = counts
        report['context'], _ = inspect_context(outputs)
        report['status'] = 'completed'
        report['control_evidence_status'] = ('zero_track_support' if not report['context']['positive_weight_count']
                                             else 'nonzero_support_requires_quality_review')
        if report['instrumentation_errors']:
            report['instrumentation_status'] = 'incomplete'
        else:
            report['instrumentation_status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(limit=8))
    finally:
        for handle in reversed(handles):
            handle.remove()
        for pending in stacks.values():
            for event in pending:
                event['phase'] = 'entered_without_return'
        report['tracking_passes'] = tracking_pass
        report['hook_calls'] = counts
        report['installed_hooks_removed'] = len(handles)
        report['elapsed_seconds_including_instrumentation'] = time.monotonic() - started
    return dict(report=report, outputs=outputs)
