"""Stage64 data contracts for learned reference-to-video edit transport.

The model inputs have no target-video field. A paired edited FIRST frame is an
explicit supervised training/monitor protocol, never an evaluation reference.
Evaluation references are recovered read-only from Stage63's legal anchors.
"""
import copy

import torch

from utils.editing_data import digest, rae_canvas


INPUT_SCHEMA = 'learned-edit-transport-input-v1'
SUPERVISION_SCHEMA = 'learned-edit-transport-supervision-v1'


def build_split(bank, cfg):
    """Keep the old split; reserve whole TRAIN source groups for monitoring.

    ``train_pairs=0`` uses all available training pairs. A positive limit is
    rounded up to complete source groups and includes monitor cases. The
    monitoring subset is distinct from the existing development set and may
    use paired first-frame references, explicitly labelled as such.
    """
    records = list(bank['records'])
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate case IDs in native bank')
    # The original manifest includes reserved tests even though they were never
    # cached. Check these too before deriving a new partition.
    assignments = {}
    identities = {}
    for row in records + list(bank.get('manifest', {}).get('records', [])):
        split = row.get('split')
        if split not in ('train', 'dev', 'test'):
            continue
        if not row.get('group'):
            raise ValueError('Need source-group identity for ' + row['id'])
        assignment = (split, row['group'])
        if row['id'] in identities and identities[row['id']] != assignment:
            raise ValueError('Case split/group changed: ' + row['id'])
        identities[row['id']] = assignment
        assignments.setdefault(row['group'], set()).add(split)
    if any(len(v) > 1 for v in assignments.values()):
        raise ValueError('Source groups overlap native train/dev/reserved-test splits')
    groups = {}
    for row in records:
        if row['split'] == 'train':
            groups.setdefault(row['group'], []).append(row)
    if not groups:
        raise ValueError('No training pairs in the reused native bank')
    training = cfg.get('training', {})
    fraction = float(training.get('monitor_fraction', .0625))
    if not 0 <= fraction < 1:
        raise ValueError('monitor_fraction must be in [0,1)')
    seed = int(cfg.get('seed', 42))
    order = sorted(groups, key=lambda g: digest(['stage64-train-selection', seed, g]))
    limit = int(cfg.get('train_pairs', 0))
    if limit < 0:
        raise ValueError('train_pairs cannot be negative')
    selected = []
    count = 0
    for group in order:
        if limit and count >= limit:
            break
        selected.append(group)
        count += len(groups[group])
    monitor_count = (min(len(selected)-1, max(1, round(len(selected)*fraction)))
                     if fraction and len(selected) > 1 else 0)
    monitor_order = sorted(selected, key=lambda g: digest(['stage64-monitor', seed, g]))
    monitor_groups = set(monitor_order[:monitor_count])
    chosen = []
    for group in selected:
        partition = 'monitor' if group in monitor_groups else 'train'
        chosen.extend(dict(r, partition=partition) for r in groups[group])
    dev = sorted((r for r in records if r['split'] == 'dev'), key=lambda r: r['id'])
    dev_limit = int(cfg.get('cases', len(dev)))
    if dev_limit < 0:
        raise ValueError('cases cannot be negative')
    chosen.extend(dict(r, partition='dev') for r in (dev[:dev_limit] if dev_limit else dev))
    chosen.sort(key=lambda r: (r['partition'], r['id']))
    counts = {p: sum(r['partition'] == p for r in chosen) for p in ('train', 'monitor', 'dev')}
    value = dict(schema='learned-edit-transport-split-v1',
        source_bank_identity=bank['identity'], seed=seed,
        monitor_fraction=fraction, requested_train_pairs=limit,
        records=chosen, counts=counts,
        protocols=dict(train='paired_first_frame_supervision',
                       monitor='heldout_train_groups_paired_first_frame_supervision',
                       dev='stage63_model_or_user_reference_no_target_condition'),
        caveat='Original byte-hash groups; re-encoded parent-scene duplicates are not certified absent.')
    value['identity'] = digest(dict(bank=bank['identity'], seed=seed,
        monitor_fraction=fraction,
        selected=[(r['id'], r['group'], r['partition']) for r in chosen]))
    return value


def _rgb(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(name + ' must be a tensor')
    if value.dtype == torch.uint8:
        result = value.detach().float().cpu()/255
    elif value.is_floating_point():
        result = value.detach().float().cpu()
        if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
            raise ValueError(name + ' float RGB must be finite in [0,1]')
    else:
        raise TypeError(name + ' RGB must be uint8 or floating point [0,1]')
    return result


def _u8(value):
    return (value.detach().float().cpu().clamp(0, 1)*255).round().byte()


def _device(model, supplied):
    if supplied is not None:
        return torch.device(supplied)
    if hasattr(model, 'parameters'):
        parameter = next(iter(model.parameters()), None)
        if parameter is not None:
            return parameter.device
    return torch.device('cpu')


def _indices(frames, count):
    if count < 2 or count > frames or (frames-1) % (count-1):
        raise ValueError('Need distinct equally-spaced source states including both endpoints')
    return list(range(0, frames, (frames-1)//(count-1)))


def prepare_inputs(source, anchor, support, spatial, cfg, rae, encoder, device=None):
    """Encode source states and one legal reference; no target-video interface.

    Returned source/proposal are uint8 [9,3,518,518]; anchor is [3,518,518].
    source_z [9,g,g,c] and anchor_z [1,g,g,c] are FP16 CPU tensors. Source
    correspondence is computed on the same nine states, not sliced from full
    frame-rate adjacent flows. Dense flows are deliberately not cached here.
    """
    from audit_edit_transport import encode_window
    from utils.editing_video_transport import propagate_reference
    source = _rgb(source, 'source')
    anchor = _rgb(anchor, 'anchor')
    if source.ndim != 4 or source.shape[1] != 3 or anchor.shape != source.shape[1:]:
        raise ValueError('Expected source [T,3,H,W] and aligned anchor [3,H,W]')
    if len(source) != int(cfg.get('frames', 17)):
        raise ValueError('Source does not have the declared frame count')
    if list(source.shape[-2:]) != list(spatial['canvas']):
        raise ValueError('Source canvas differs from native spatial metadata')
    top, left, height, width = map(int, spatial['valid_rect'])
    if min(top, left) < 0 or min(height, width) <= 0 or top+height > source.shape[-2] or left+width > source.shape[-1]:
        raise ValueError('Invalid native valid rectangle')
    support = torch.as_tensor(support).detach().float().cpu()
    if support.shape != (1, *source.shape[-2:]) or not bool(torch.isfinite(support).all()):
        raise ValueError('Need finite first-frame support [1,H,W]')
    if bool(((support < 0) | (support > 1)).any()):
        raise ValueError('Support must lie in [0,1]')
    indices = _indices(len(source), int(cfg.get('rae_frames', 9)))
    square, rect = rae_canvas(source[indices], spatial['valid_rect'])
    square_anchor, anchor_rect = rae_canvas(anchor[None], spatial['valid_rect'])
    square_support, support_rect = rae_canvas(support[None], spatial['valid_rect'])
    if rect != anchor_rect or rect != support_rect:
        raise ValueError('Source/reference/support letterbox coordinates differ')
    square_support = square_support[0].clamp(0, 1)
    flow = propagate_reference(square, square_anchor[0], square_support, cfg.get('transport', {}))
    # The observed first frame is a boundary condition. Interpolating a mask
    # can produce soft edge values; multiplying the already-edited reference
    # by these a second time must not weaken its known effect.
    flow['proposal'][0] = square_anchor[0]
    actual_device = _device(rae, device)
    z = encode_window(rae, encoder, square[None].to(actual_device)).detach().float()
    # Independent first-frame encoding does not consume future edited frames.
    za = encode_window(rae, encoder, square_anchor[None].to(actual_device)).detach().float()
    if z.ndim != 5 or za.ndim != 5 or z.shape[:2] != (1, len(indices)) or za.shape[:2] != (1, 1):
        raise ValueError('Encoder returned an incompatible source/reference state layout')
    if z.shape[2:] != za.shape[2:] or not bool(torch.isfinite(z).all() and torch.isfinite(za).all()):
        raise ValueError('Source/reference latent dimensions or values differ')
    return dict(schema=INPUT_SCHEMA, source=_u8(square), proposal=_u8(flow['proposal']),
        anchor=_u8(square_anchor[0]), support=square_support.to(torch.float16),
        confidence=flow['confidence'].detach().cpu().to(torch.float16),
        masks=flow['masks'].detach().cpu().to(torch.float16),
        source_z=z[0].cpu().to(torch.float16), anchor_z=za[0].cpu().to(torch.float16),
        observed_indices=indices, valid_rect=rect, source_spatial=copy.deepcopy(spatial),
        source_frames=len(source), flow_metadata=copy.deepcopy(flow['metadata']),
        protocol='source_plus_declared_first_frame_reference')


def prepare_training_pair(sample, cfg, rae, encoder, device=None, partition='train'):
    """Build target-free input first, then a separate future supervision payload.

    ``monitor`` is held out from the old training split, not the public/dev
    inference protocol. Using target[0] here is declared first-frame-reference
    supervision; calling this on dev/test is an error.
    """
    from edit_reference_video import source_support
    if partition not in ('train', 'monitor'):
        raise ValueError('Paired first-frame references are training/monitor only')
    row = sample.get('row', {})
    if row.get('split', 'train') != 'train':
        raise ValueError('Do not use dataset dev/test references as model conditions')
    source = _rgb(sample['source'], 'source')
    # Read ONLY this target frame until the input object is fully constructed.
    anchor = _rgb(sample['target'][:1], 'training first-frame reference')[0]
    spatial = sample['timing']['spatial']
    support, diagnostics = source_support(source, anchor, spatial, None, cfg)
    anchor = source[0]*(1-support)+anchor*support
    inputs = prepare_inputs(source, anchor, support, spatial, cfg, rae, encoder, device)
    inputs['protocol'] = ('paired_first_frame_supervision' if partition == 'train'
                          else 'heldout_train_groups_paired_first_frame_supervision')
    inputs['reference_provenance'] = 'paired_training_target_first_frame'
    indices = inputs['observed_indices']
    target = _rgb(sample['target'][indices], 'supervision target')
    if target.shape != source[indices].shape:
        raise ValueError('Paired target/source canvas differs')
    target_square, target_rect = rae_canvas(target, spatial['valid_rect'])
    if target_rect != inputs['valid_rect']:
        raise ValueError('Supervision letterbox coordinates differ')
    supervision = dict(schema=SUPERVISION_SCHEMA, target=_u8(target_square),
        observed_indices=indices, valid_rect=target_rect,
        protocol='paired_future_video_supervision_not_condition')
    return inputs, supervision, dict(diagnostics, partition=partition,
        reference_protocol=inputs['protocol'], future_target_conditioned=False)


def prepare_stage63_case(anchors_store, case_id, cfg, rae, encoder, device=None):
    """Read legal Stage63 reference artifacts, never a dataset target record."""
    manifest = anchors_store.json('manifest.json', True)
    if case_id not in {row['id'] for row in manifest['cases']}:
        raise ValueError('Case is absent from the Stage63 reference manifest')
    anchor = anchors_store.load('anchors/'+case_id+'/result.pt', manifest['identity'], True)
    if anchor['provenance'] not in ('model_output', 'user_edit'):
        raise ValueError('Development reference must have declared legal provenance')
    result = prepare_inputs(anchor['source'], anchor['anchor'], anchor['support'],
        anchor['spatial'], cfg, rae, encoder, device)
    result.update(protocol='stage63_model_or_user_reference_no_target_condition',
        reference_provenance=anchor['provenance'], stage63_identity=manifest['identity'],
        case_id=case_id)
    return result
