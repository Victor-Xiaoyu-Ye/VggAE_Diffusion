#!/usr/bin/env python3
"""Diagnose camera-compensated VGGT track masks on saved fixed-window previews.

No diffusion/AE training. Full-window teacher masks are future-assisted labels;
observed-prefix masks are computed in a SEPARATE forward on observed RGB only.
One independent worker per node, with per-case commits and no collectives.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np

from utils.file_signature import sampled_file_signature
from utils.native_audit_io import recover_case, seal, sha256
from utils.scene_run import ArtifactStore, atomic, digest
from utils.scene_trackmask import TrackMaskConfig, camera_compensated_track_mask
from utils.scene_trackmask_audit import (INPUT_NOTE, MASK_NOTE, decode_preview,
    input_contact_sheet, point_support, read_case, region_metrics, render_mask_review)


class TeacherSetupError(RuntimeError):
    """An incompatible/missing teacher is a setup error, not 64 bad videos."""


def case_indices(value):
    result = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise ValueError('cases must be nonempty unique nonnegative indices')
    return result


@contextmanager
def heartbeat(phase, case):
    stop = threading.Event()
    started = time.monotonic()
    def report():
        while not stop.wait(30):
            print('[trackmask heartbeat] '+json.dumps(dict(phase=phase, case=case,
                elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def archive_arrays(path, values):
    """Atomic numeric-only NPZ: safe to inspect without arbitrary-code unpickling."""
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{key: value for key, value in values.items()
                                      if isinstance(value, np.ndarray)})
    os.replace(temporary, path)


def classify(teacher, config):
    return camera_compensated_track_mask(tracks=teacher['tracks'], depth=teacher['depth'],
        world_to_camera=teacher['extrinsics'], intrinsics=teacher['intrinsics'],
        visibility=teacher['visibility'], track_confidence=teacher['track_confidence'],
        depth_confidence=teacher['depth_conf'], reference_frame=0, config=config)


def threshold_sensitivity(teacher, config):
    rows = []
    for threshold, fraction in ((1., .5), (2., .5), (4., .5), (2., .25)):
        candidate = replace(config, base_threshold_px=threshold, min_dynamic_fraction=fraction)
        result = classify(teacher, candidate)
        rows.append(dict(base_threshold_px=threshold, min_dynamic_fraction=fraction,
                         **result['diagnostics']))
    return rows


def checkpoint_identity(args):
    root = Path(args.teacher_code).resolve()
    files = sorted(root.rglob('*.py'))
    if not files:
        raise FileNotFoundError(f'No full-head VGGT code in {root}')
    return dict(checkpoint=sampled_file_signature(args.teacher_checkpoint),
                code_sha256=digest({file.relative_to(root).as_posix(): sha256(file) for file in files}),
                precision=args.precision, query_stride=args.query_stride,
                device_type=args.device.split(':')[0],
                code_root=str(root), checkpoint_path=str(Path(args.teacher_checkpoint).resolve()))


def obtain_teacher(store, directory, raw, history, identity, teacher_factory):
    """Commit expensive inference before analysis/render, recover by content hash."""
    receipt_name = directory+'/teacher.done.json'
    try:
        receipt = recover_case(store.local, receipt_name, identity, store.roots)
    except (OSError, RuntimeError) as exc:
        print(f'[trackmask] repairing teacher payload: {exc}', flush=True)
        receipt = None
    if receipt:
        store.publish([*receipt['files'], receipt_name])
        values = []
        for label in ('full', 'prefix'):
            with np.load(store.local/directory/f'{label}_teacher.npz', allow_pickle=False) as data:
                values.append({key: data[key] for key in data.files})
        return values, 0
    teacher = teacher_factory()
    frames = np.ascontiguousarray(raw.transpose(0, 3, 1, 2).astype(np.float32)/255.)
    values = []
    descriptions = []
    files = []
    for label, images in (('full', frames), ('prefix', frames[:history])):
        with heartbeat('teacher_'+label, directory):
            try:
                result = teacher.infer(images, reference_frame=0)
            except (RuntimeError, NotImplementedError) as exc:
                # A fixed-shape device/operator failure will not be repaired
                # by trying another 63 videos with the same configuration.
                text = str(exc).lower()
                if isinstance(exc, NotImplementedError) or any(marker in text for marker in (
                        'out of memory', 'not implemented', 'unsupported', 'does not support')):
                    raise TeacherSetupError(f'Teacher runtime cannot execute this configuration: {exc}') from exc
                raise
        result = {key: value.detach().cpu().numpy() if hasattr(value, 'detach') else value
                  for key, value in result.items()}
        file = directory+f'/{label}_teacher.npz'
        archive_arrays(store.local/file, result)
        files.append(file)
        descriptions.append({key: value for key, value in result.items()
                             if not isinstance(value, np.ndarray)})
        values.append(result)
    metadata = dict(full_context='All raw frames, GT-assisted diagnostic only',
        prefix_context=f'Only raw[:{history}], a separate forward with no future input',
        teacher=descriptions)
    file = directory+'/teacher.json'
    atomic(store.local/file, metadata)
    files.append(file)
    seal(store.local, receipt_name, identity, files, metadata)
    store.publish([*files, receipt_name])
    return values, len(frames)+history


def collect(store, nodes, *, write_partial=False):
    reports, missing = [], []
    for rank in range(nodes):
        try:
            report = store.json(f'nodes/node{rank:03d}.json')
        except (OSError, RuntimeError) as exc:
            missing.append(dict(node=rank, error=repr(exc)))
            continue
        if report is None:
            missing.append(dict(node=rank, error='not yet published'))
        else:
            reports.append(report)
    contracts = {row['contract_hash'] for row in reports}
    if len(contracts) > 1:
        raise ValueError('node reports belong to different audit contracts')
    all_rows = [case for report in reports for case in report['cases']]
    by_case = {row['case_index']: row for row in all_rows}
    if len(by_case) != len(all_rows):
        raise ValueError('case ownership duplicated across node reports')
    prepared = sum(row['status'] == 'prepared_only' for row in all_rows)
    completed = sum(row['status'] == 'completed' for row in all_rows)
    failed = sum(row['status'] == 'failed' for row in all_rows)
    expected = reports[0]['requested_cases'] if reports else []
    coverage_complete = set(by_case) == set(expected) and not missing
    summary = dict(schema='scene-trackmask-summary-v1', nodes=nodes, missing_nodes=missing,
        completed=completed, prepared_only=prepared, failed=failed, expected=len(expected),
        status=('partial' if not coverage_complete else 'completed_with_errors' if failed
                else 'prepared_only' if prepared else 'completed'),
        teacher_quality_validated=False,
        interpretation='Requires visual inspection; completion does not establish segmentation or generation quality.',
        input_note=INPUT_NOTE, mask_note=MASK_NOTE, cases=sorted(all_rows, key=lambda row: row['case_index']))
    if coverage_complete or write_partial:
        store.write_json('summary.json' if coverage_complete else 'collection_partial.json', summary)
    return summary


def run(args):
    if not 0 <= args.node_rank < args.nodes or args.query_stride < 1 or args.allocated_devices < 1:
        raise ValueError('invalid node topology or query stride')
    os.environ['RANK'] = str(args.node_rank)  # Unique publication status filenames; no torchrun.
    store = ArtifactStore(args.output, args.root)
    if args.collect_only:
        result = collect(store, args.nodes, write_partial=True)
        print(json.dumps({key: value for key, value in result.items() if key != 'cases'}), flush=True)
        return 0
    if not args.source_local:
        raise ValueError('--source-local is required')
    source = Path(args.source_local).resolve()
    destination = store.local.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and audit output directories must be disjoint')
    if set(args.source_root) & set(args.root):
        raise ValueError('source/output remote roots must differ')
    requested = case_indices(args.cases)
    assigned = requested[args.node_rank::args.nodes]
    cfg = TrackMaskConfig()
    teacher_spec = None if args.prepare_only else checkpoint_identity(args)
    contract = dict(schema='scene-trackmask-contract-v1', requested_cases=requested,
        nodes=args.nodes,
        source_step=args.step, source_seed=args.seed, source_weights=args.weights,
        teacher=teacher_spec, geometry=asdict(cfg), prepare_only=args.prepare_only,
        input_note=INPUT_NOTE, mask_note=MASK_NOTE,
        point_support_radius_px=args.query_stride/2.,
        condition_policy='teacher_full_window_for_analysis_only; prefix_recomputed_separately')
    repo = Path(__file__).parent
    contract['audit_code'] = {name: sha256(repo/name) for name in (
        'audit_scene_trackmask.py', 'utils/scene_trackmask.py', 'utils/scene_trackmask_audit.py',
        'utils/scene_trackmask_teacher.py')}
    contract_hash = digest(contract)
    contract_file = f'nodes/contract{args.node_rank:03d}.json'
    old = store.json(contract_file)
    if old and old != contract:
        raise ValueError('output contains another contract; select a different audit namespace')
    store.write_json(contract_file, contract)
    teacher_instance = None
    def teacher_factory():
        nonlocal teacher_instance
        if teacher_instance is None:
            from utils.scene_trackmask_teacher import VGGTTrackTeacher
            with heartbeat('loading_teacher', args.node_rank):
                try:
                    teacher_instance = VGGTTrackTeacher(args.teacher_checkpoint,
                        code_root=args.teacher_code, device=args.device,
                        precision=args.precision, query_stride=args.query_stride)
                except Exception as exc:
                    raise TeacherSetupError(f'Cannot initialize compatible full-head teacher: {exc}') from exc
        return teacher_instance
    all_rows = []
    started = time.monotonic()
    inference_frames = 0
    for index in assigned:
        case_start = time.monotonic()
        directory = f'cases/case{index:03d}'
        (store.local/directory).mkdir(parents=True, exist_ok=True)
        try:
            path, metadata, source_identity = read_case(source, args.source_root, index,
                step=args.step, seed=args.seed, weights=args.weights)
            identity = digest(dict(contract=contract_hash, source=source_identity))
            receipt_name = directory+'/done.json'
            try:
                receipt = recover_case(store.local, receipt_name, identity, store.roots)
            except (OSError, RuntimeError) as exc:
                print(f'[trackmask] repair case{index:03d}: {exc}', flush=True)
                receipt = None
            if receipt:
                store.publish([*receipt['files'], receipt_name])
                all_rows.append(receipt['metadata'])
                print(f'[trackmask] recovered case{index:03d}', flush=True)
                continue
            columns = decode_preview(path, metadata)
            history = metadata['history_frames']
            row = dict(case_index=index, id=metadata['id'], frame_ids=metadata['frame_ids'],
                history_frames=history, source=source_identity, source_video=str(path),
                input_note=INPUT_NOTE, status='prepared_only' if args.prepare_only else 'completed')
            input_contact_sheet(columns, history, store.local/directory/'input_review.jpg')
            files = [directory+'/input_review.jpg']
            if not args.prepare_only:
                (full_teacher, prefix_teacher), frames = obtain_teacher(store, directory,
                    columns['RAW'], history, identity, teacher_factory)
                inference_frames += frames
                full = classify(full_teacher, cfg)
                prefix = classify(prefix_teacher, cfg)
                # Geometry labels use all supplied frames, but this second call
                # has only observed frames. Never copy full labels into prefix.
                prefix['diagnostics']['uses_future_frames'] = False
                prefix['diagnostics']['scope'] = 'observed prefix only; does not predict future masks'
                height, width = columns['RAW'].shape[1:3]
                support = point_support(full_teacher['tracks'], full['labels'], height, width,
                                        radius=args.query_stride/2.)
                row.update(full=full['diagnostics'], prefix=prefix['diagnostics'],
                    regions=region_metrics(columns, support, history),
                    sensitivity=threshold_sensitivity(full_teacher, cfg), mask_note=MASK_NOTE)
                for label, result in (('full', full), ('prefix', prefix)):
                    name = directory+f'/{label}_mask.npz'
                    archive_arrays(store.local/name, result)
                    files.append(name)
                names = render_mask_review(columns, dict(full, tracks=full_teacher['tracks']),
                    dict(prefix, tracks=prefix_teacher['tracks']), store.local/directory, history,
                    fps=metadata.get('nominal_fps', 8.))
                files += [directory+'/'+name for name in names]
            row['case_seconds'] = time.monotonic()-case_start
            file = directory+'/metrics.json'
            atomic(store.local/file, row)
            files.append(file)
            seal(store.local, receipt_name, identity, files, row)
            store.publish([*files, receipt_name])
            all_rows.append(row)
        except Exception as exc:
            if isinstance(exc, TeacherSetupError):
                raise
            traceback.print_exc()
            row = dict(case_index=index, status='failed', error=repr(exc))
            all_rows.append(row)
            store.write_json(directory+'/error.json', row)
        elapsed = max(time.monotonic()-started, 1e-6)
        progress = dict(phase='prepare' if args.prepare_only else 'trackmask', node=args.node_rank,
            processed=len(all_rows), assigned=len(assigned), case_index=index,
            status=all_rows[-1]['status'], elapsed_seconds=elapsed,
            inferred_frames=inference_frames, DI_throughput=inference_frames/elapsed,
            DI_unit='teacher input frames/s/active worker; full+prefix counted separately',
            allocated_devices=args.allocated_devices,
            DI_throughput_per_allocated_device=inference_frames/(elapsed*args.allocated_devices),
            DI_note='End-to-end elapsed includes IO/render/publication; no guard work or resumed frames counted.')
        print('[trackmask] '+json.dumps(progress), flush=True)
        store.write_json(f'nodes/progress{args.node_rank:03d}.json', progress)
    report = dict(contract_hash=contract_hash, requested_cases=requested, assigned_cases=assigned,
        node=args.node_rank, nodes=args.nodes, cases=all_rows,
        elapsed_seconds=time.monotonic()-started, inferred_frames=inference_frames,
        status='completed_with_errors' if any(row['status'] == 'failed' for row in all_rows)
        else 'prepared_only' if args.prepare_only else 'completed')
    store.write_json(f'nodes/node{args.node_rank:03d}.json', report)
    collect(store, args.nodes)  # Best effort; never wait for another node or overwrite a complete report with partial.
    return 1 if any(row['status'] == 'failed' for row in all_rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-local', default='')
    parser.add_argument('--source-root', action='append', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', action='append', default=[])
    parser.add_argument('--teacher-code', default=str(Path(__file__).parent/'third_party/vggt_trackmask'))
    parser.add_argument('--teacher-checkpoint', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--query-stride', type=int, default=21)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--allocated-devices', type=int, default=1)
    parser.add_argument('--cases', default=','.join(map(str, range(64))))
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--step', type=int, default=6000)
    parser.add_argument('--weights', choices=('online', 'ema'), default='ema')
    parser.add_argument('--prepare-only', action='store_true', help='Verify inputs and export comparison sheets; no teacher inference')
    parser.add_argument('--collect-only', action='store_true', help='Collect completed per-node reports without running any model')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
