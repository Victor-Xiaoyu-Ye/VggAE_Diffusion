"""Content-checked, per-case resume for inference audits (no optimizer state)."""
import hashlib
import json
import os
from pathlib import Path

from scripts.window_run_io import child, atomic
from utils.moxing_io import copy_file, is_remote_path, read_text


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exists(path):
    if is_remote_path(path):
        import moxing
        return moxing.file.exists(path)  # Never turn an access error into "absent".
    return Path(path).exists()


def safe_relative(name):
    from pathlib import PurePosixPath
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('unsafe artifact path')
    return name


def recover_json(name, out, roots, required=False):
    name = safe_relative(name)
    errors = []
    for root in [str(out), *roots]:
        path = child(root, name)
        try:
            if exists(path):
                return json.loads(read_text(path))
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    if errors or required:
        raise RuntimeError(f'Cannot recover {name}: {errors or "not found"}')
    return None


def seal(out, receipt_name, contract_hash, files, metadata):
    out = Path(out)
    receipt = dict(schema='native-i2v-case-v1', status='completed', contract_hash=contract_hash,
        files={safe_relative(name): dict(size=(out/name).stat().st_size, sha256=sha256(out/name))
               for name in files}, metadata=metadata)
    atomic(out / receipt_name, receipt)
    return receipt


def recover_case(out, name, contract_hash, roots):
    receipt = recover_json(name, out, roots)
    if receipt is None:
        return None
    if (receipt.get('schema') != 'native-i2v-case-v1' or receipt.get('status') != 'completed'
            or receipt.get('contract_hash') != contract_hash or not receipt.get('files')):
        raise ValueError(f'incompatible receipt: {name}')
    out = Path(out)
    for relative, expected in receipt['files'].items():
        target = out / safe_relative(relative)
        def valid(p):
            return p.is_file() and p.stat().st_size == expected['size'] and sha256(p) == expected['sha256']
        if valid(target):
            continue
        errors = []
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        for root in roots:
            try:
                copy_file(child(root, relative), str(partial))
                if not valid(partial):
                    raise ValueError('content hash/size mismatch')
                os.replace(partial, target)
                break
            except Exception as exc:
                errors.append(repr(exc))
                partial.unlink(missing_ok=True)
        else:
            raise RuntimeError(f'incomplete committed case {name}/{relative}: {errors}')
    atomic(out/name, receipt)
    return receipt


def validate_wan_checkpoint(root):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    expected = dict(model_type='i2v', dim=5120, num_layers=40, num_heads=40, in_dim=36, out_dim=16)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Expected native Wan2.1 I2V-14B {key}={value}, got {cfg.get(key)}')
    index = root/'diffusion_pytorch_model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        names.append(index.name)
    else:
        names = ['diffusion_pytorch_model.safetensors']
    names += ['config.json', 'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
              'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']
    for folder in ('google/umt5-xxl', 'xlm-roberta-large'):
        token_dir = root / folder
        if not token_dir.is_dir() or not any((token_dir/name).is_file()
                for name in ('tokenizer.json', 'spiece.model', 'sentencepiece.bpe.model')):
            raise FileNotFoundError(f'missing local tokenizer assets: {token_dir}')
        names.extend(p.relative_to(root).as_posix() for p in token_dir.rglob('*') if p.is_file())
    from utils.file_signature import sampled_file_signature
    signatures = {}
    for name in sorted(set(names)):
        path = root / safe_relative(name)
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'missing/empty native checkpoint component: {path}')
        signatures[name] = sampled_file_signature(str(path))
    return dict(config=cfg, files=signatures)


def aligned_indices(frame_num, fps=16, duration=1., count=9):
    # Native FPS matches the trained pipeline. Do not time-stretch the full clip.
    indices = [round(i * duration * fps / (count-1)) for i in range(count)]
    if len(set(indices)) != count or indices[-1] >= frame_num:
        raise ValueError('native output cannot provide the requested one-second window')
    return indices
