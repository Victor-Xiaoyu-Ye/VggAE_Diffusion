"""Strict construction, loading, and contracts for the R7 representation.

The representation is deliberately defined without filesystem paths.  Contracts may
carry inexpensive sampled signatures, but never depend on where an artifact was
staged on a worker.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from data.token_utils import strip_special_tokens
from models.causal_dual_tokenizer import CausalDualTokenizerCore
from models.dpt_latent_decoder import CompactCompressor
from models.dual_stream_decoder import DualStreamDecoder
from models.texture_encoder import TextureEncoder
from utils.file_signature import sampled_file_signature

R7_CONTRACT_SCHEMA = "r7-representation-v1"
REQUIRED_PREFIXES = ("compressor.", "tex_encoder.", "tokenizer.", "decoder.")
SOURCE_PREFIXES = ("compressor.", "tex_encoder.", "decoder.")


def checkpoint_args(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Return checkpoint args as a plain dictionary."""
    value = checkpoint.get("args", {})
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint 'args' must be a mapping or Namespace")
    return dict(value)


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    return checkpoint


def model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint must contain a non-empty 'model' state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError("checkpoint 'model' is not a tensor state dict")
    return state


def prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str, *, required: bool = True,
) -> Dict[str, torch.Tensor]:
    result = {key[len(prefix):]: value for key, value in state.items()
              if key.startswith(prefix)}
    if required and not result:
        raise RuntimeError(f"checkpoint lacks required {prefix}* keys")
    return result


def validate_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: Sequence[str] = REQUIRED_PREFIXES,
) -> None:
    missing = [prefix for prefix in prefixes if not any(
        key.startswith(prefix) for key in state)]
    if missing:
        raise RuntimeError(f"checkpoint lacks required prefixes: {missing}")


def load_prefixed_strict(
    module: nn.Module, state: Mapping[str, torch.Tensor], prefix: str,
) -> int:
    """Strictly load one namespaced module and return the matched key count."""
    substate = prefixed_state(state, prefix)
    expected = module.state_dict()
    missing = sorted(set(expected) - set(substate))
    unexpected = sorted(set(substate) - set(expected))
    shape_mismatch = sorted(
        key for key in set(expected).intersection(substate)
        if tuple(expected[key].shape) != tuple(substate[key].shape))
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"strict {prefix} load failed: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}")
    module.load_state_dict(substate, strict=True)
    return len(substate)


def _has_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


def _infer_resblock_count(state: Mapping[str, torch.Tensor]) -> int:
    indices = set()
    prefix = "decoder.up0.resblocks."
    for key in state:
        if not key.startswith(prefix):
            continue
        piece = key[len(prefix):].split(".", 1)[0]
        if piece.isdigit():
            indices.add(int(piece))
    return max(indices) + 1 if indices else 2


def infer_decoder_temporal_blocks(state: Mapping[str, torch.Tensor]) -> int:
    if _has_prefix(state, "decoder.temporal_mid."):
        return 2
    if _has_prefix(state, "decoder.temporal_low."):
        return 1
    return 0


def _infer_tokenizer_config(
    state: Mapping[str, torch.Tensor], values: Mapping[str, Any],
) -> Dict[str, int]:
    geo_weight = state.get("tokenizer.geo_projection.compress.weight")
    tex_weight = state.get("tokenizer.tex_projection.compress.weight")
    if geo_weight is None or tex_weight is None:
        return {
            "geo_dim": int(values.get("geo_dim", 256)),
            "tex_dim": int(values.get("tex_dim", 256)),
            "geo_latent_dim": int(values.get("geo_latent_dim", 96)),
            "tex_latent_dim": int(values.get("tex_latent_dim", 96)),
        }
    return {
        "geo_dim": int(geo_weight.shape[1]),
        "tex_dim": int(tex_weight.shape[1]),
        "geo_latent_dim": int(geo_weight.shape[0]),
        "tex_latent_dim": int(tex_weight.shape[0]),
    }


def _infer_temporal_depth(state: Mapping[str, torch.Tensor]) -> int:
    indices = set()
    prefix = "tokenizer.temporal.encoder.blocks."
    for key in state:
        if key.startswith(prefix):
            piece = key[len(prefix):].split(".", 1)[0]
            if piece.isdigit():
                indices.add(int(piece))
    return max(indices) + 1 if indices else 3


def _infer_temporal_factor(
    state: Mapping[str, torch.Tensor], values: Mapping[str, Any],
) -> int:
    weight = state.get("tokenizer.temporal.encoder.fold.weight")
    return int(weight.shape[2]) if weight is not None else int(
        values.get("temporal_factor", 2))


@dataclass(frozen=True)
class R7Config:
    target_size: int = 518
    latent_grid: int = 18
    input_grid: int = 37
    levels: Tuple[int, ...] = (4, 11, 17, 23)
    token_dim: int = 2048
    geo_dim: int = 256
    tex_dim: int = 256
    tex_base_ch: int = 64
    tex_pack: str = "avgpool"
    decoder_base_dim: int = 384
    decoder_num_resblocks: int = 2
    decoder_temporal_blocks: int = 0
    decoder_use_checkpoint: bool = True
    temporal_factor: int = 2
    temporal_depth: int = 3
    geo_latent_dim: int = 96
    tex_latent_dim: int = 96
    seq_len: int = 9
    clip_duration_seconds: float = 1.0

    @property
    def latent_dim(self) -> int:
        return self.geo_latent_dim + self.tex_latent_dim

    @property
    def latent_seq_len(self) -> int:
        return 1 + (self.seq_len - 1) // self.temporal_factor

    def validate(self) -> None:
        if self.temporal_factor not in (2, 4):
            raise ValueError(f"temporal_factor must be 2 or 4, got {self.temporal_factor}")
        if self.seq_len < 2 or (self.seq_len - 1) % self.temporal_factor:
            raise ValueError(
                f"seq_len={self.seq_len} must satisfy 1 + "
                f"temporal_factor({self.temporal_factor})*k")
        if self.input_grid != self.target_size // 14:
            raise ValueError(
                f"input_grid={self.input_grid} != target_size//14="
                f"{self.target_size // 14}")
        for name in ("latent_grid", "geo_dim", "tex_dim", "geo_latent_dim",
                     "tex_latent_dim", "temporal_depth", "decoder_num_resblocks"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.decoder_temporal_blocks not in (0, 1, 2):
            raise ValueError("decoder_temporal_blocks must be 0, 1, or 2")

    @classmethod
    def from_args(
        cls, values: Mapping[str, Any], *, state: Optional[Mapping[str, torch.Tensor]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "R7Config":
        original_values = dict(values)
        values = dict(original_values)
        if overrides:
            values.update({key: value for key, value in overrides.items()
                           if value is not None})
        target_size = int(values.get("target_size", values.get("img_size", 518)))
        inferred_tokenizer = (_infer_tokenizer_config(state, values)
                              if state is not None else {})
        decoder_temporal = values.get("decoder_temporal_blocks",
                                      values.get("num_temporal_blocks"))
        if state is not None:
            # State is authoritative; this also handles the legacy step-3000 R7
            # artifact whose args inherited a source-decoder value it did not save.
            decoder_temporal = infer_decoder_temporal_blocks(state)
        config = cls(
            target_size=target_size,
            latent_grid=int(values.get("latent_grid", 18)),
            input_grid=int(values.get("input_grid", target_size // 14)),
            levels=tuple(int(x) for x in values.get("levels", (4, 11, 17, 23))),
            token_dim=int(values.get("token_dim", 2048)),
            geo_dim=int(inferred_tokenizer.get("geo_dim", values.get("geo_dim", 256))),
            tex_dim=int(inferred_tokenizer.get("tex_dim", values.get("tex_dim", 256))),
            tex_base_ch=int(values.get("tex_base_ch", 64)),
            tex_pack=str(values.get("tex_pack", "avgpool")),
            decoder_base_dim=int(values.get("decoder_base_dim", 384)),
            decoder_num_resblocks=int(
                _infer_resblock_count(state) if state is not None else
                values.get("decoder_num_resblocks", values.get("num_resblocks", 2))),
            decoder_temporal_blocks=int(decoder_temporal or 0),
            decoder_use_checkpoint=bool(values.get("use_checkpoint", True)),
            temporal_factor=int(
                _infer_temporal_factor(state, values) if state is not None else
                values.get("temporal_factor", 2)),
            temporal_depth=int(
                _infer_temporal_depth(state) if state is not None else
                values.get("temporal_depth", 3)),
            geo_latent_dim=int(inferred_tokenizer.get(
                "geo_latent_dim", values.get("geo_latent_dim", 96))),
            tex_latent_dim=int(inferred_tokenizer.get(
                "tex_latent_dim", values.get("tex_latent_dim", 96))),
            seq_len=int(values.get("seq_len", 9)),
            clip_duration_seconds=float(values.get("clip_duration_seconds", 1.0)),
        )
        config.validate()
        latent_dim = values.get("latent_dim")
        if latent_dim is not None and int(latent_dim) != config.latent_dim:
            raise ValueError(
                f"checkpoint latent_dim={latent_dim} != split total {config.latent_dim}")
        if state is not None:
            checks = {
                "geo_dim": config.geo_dim,
                "tex_dim": config.tex_dim,
                "geo_latent_dim": config.geo_latent_dim,
                "tex_latent_dim": config.tex_latent_dim,
                "temporal_factor": config.temporal_factor,
                "temporal_depth": config.temporal_depth,
            }
            for name, inferred in checks.items():
                if name in original_values and int(original_values[name]) != inferred:
                    raise ValueError(
                        f"checkpoint args {name}={original_values[name]} != "
                        f"state-inferred {inferred}")
        return config


def build_modules(config: R7Config, *, include_tokenizer: bool = True):
    """Construct source dual-AE modules and optionally the R7 tokenizer."""
    config.validate()
    compressor = CompactCompressor(
        levels=config.levels, token_dim=config.token_dim, cdim=config.geo_dim,
        latent_grid=config.latent_grid, input_grid=config.input_grid)
    tex_encoder = TextureEncoder(
        out_dim=config.tex_dim, out_grid=config.latent_grid,
        base_ch=config.tex_base_ch, img_size=config.target_size,
        pack_mode=config.tex_pack)
    decoder = DualStreamDecoder(
        geo_dim=config.geo_dim, tex_dim=config.tex_dim,
        base_dim=config.decoder_base_dim, img_size=config.target_size,
        latent_grid=config.latent_grid,
        num_resblocks=config.decoder_num_resblocks,
        num_temporal_blocks=config.decoder_temporal_blocks,
        use_checkpoint=config.decoder_use_checkpoint)
    tokenizer = None
    if include_tokenizer:
        tokenizer = CausalDualTokenizerCore(
            config.geo_dim, config.tex_dim, config.geo_latent_dim,
            config.tex_latent_dim, config.temporal_factor,
            config.temporal_depth)
    return compressor, tex_encoder, tokenizer, decoder


def load_source_modules(
    checkpoint: Mapping[str, Any], *, overrides: Optional[Mapping[str, Any]] = None,
):
    state = model_state(checkpoint)
    validate_prefixes(state, SOURCE_PREFIXES)
    config = R7Config.from_args(checkpoint_args(checkpoint), state=state,
                                overrides=overrides)
    compressor, tex_encoder, _, decoder = build_modules(
        config, include_tokenizer=False)
    matched = {
        "compressor": load_prefixed_strict(compressor, state, "compressor."),
        "tex_encoder": load_prefixed_strict(tex_encoder, state, "tex_encoder."),
        "decoder": load_prefixed_strict(decoder, state, "decoder."),
    }
    return config, compressor, tex_encoder, decoder, matched


def load_r7_modules(
    checkpoint: Mapping[str, Any], *, overrides: Optional[Mapping[str, Any]] = None,
):
    state = model_state(checkpoint)
    validate_prefixes(state, REQUIRED_PREFIXES)
    config = R7Config.from_args(checkpoint_args(checkpoint), state=state,
                                overrides=overrides)
    compressor, tex_encoder, tokenizer, decoder = build_modules(config)
    modules = (compressor, tex_encoder, tokenizer, decoder)
    matched = {}
    for module, prefix in zip(modules, REQUIRED_PREFIXES):
        matched[prefix[:-1]] = load_prefixed_strict(module, state, prefix)
    return config, compressor, tex_encoder, tokenizer, decoder, matched


def load_r7_tokenizer(
    checkpoint: Mapping[str, Any], *, overrides: Optional[Mapping[str, Any]] = None,
):
    """Strictly construct only the tokenizer for non-preview diffusion ranks."""
    state = model_state(checkpoint)
    validate_prefixes(state, REQUIRED_PREFIXES)
    config = R7Config.from_args(checkpoint_args(checkpoint), state=state,
                                overrides=overrides)
    tokenizer = CausalDualTokenizerCore(
        config.geo_dim, config.tex_dim, config.geo_latent_dim,
        config.tex_latent_dim, config.temporal_factor, config.temporal_depth)
    matched = load_prefixed_strict(tokenizer, state, "tokenizer.")
    return config, tokenizer, matched


def merged_representation_state(
    compressor: nn.Module, tex_encoder: nn.Module, tokenizer: nn.Module,
    decoder: nn.Module,
) -> Dict[str, torch.Tensor]:
    merged = {}
    for module, prefix in zip(
            (compressor, tex_encoder, tokenizer, decoder), REQUIRED_PREFIXES):
        merged.update({prefix + key: value.detach().cpu()
                       for key, value in module.state_dict().items()})
    validate_prefixes(merged)
    return merged


def validate_source_artifact(
    checkpoint: Mapping[str, Any], source_path: str,
) -> Dict[str, Any]:
    """Require the supplied source dual-AE to match frozen R7 source weights."""
    r7_state = model_state(checkpoint)
    source = load_checkpoint(source_path)
    source_state = model_state(source)
    # Only the permanently frozen modules are compared. The decoder is a
    # training target in PHASE=joint, so a joint checkpoint legitimately no
    # longer matches the source decoder and requiring it to match would make
    # joint runs impossible to resume. Whole-artifact identity is still
    # guarded by contract["signatures"]["source_dual_ae"].
    frozen_prefixes = ("compressor.", "tex_encoder.")
    expected = {
        key: value for key, value in r7_state.items()
        if any(key.startswith(prefix) for prefix in frozen_prefixes)}
    missing = sorted(set(expected) - set(source_state))
    mismatched = sorted(
        key for key, value in expected.items()
        if key in source_state and (
            tuple(value.shape) != tuple(source_state[key].shape)
            or value.dtype != source_state[key].dtype
            or not torch.equal(value.cpu(), source_state[key].cpu())))
    if missing or mismatched:
        raise ValueError(
            "--dual_ae_ckpt does not match the frozen compressor/texture "
            "weights embedded in the R7 checkpoint: "
            f"missing={missing[:8]}, mismatched={mismatched[:8]}")
    # Additional source-only keys are allowed: the legacy R7 probe deliberately
    # instantiated a decoder without optional temporal blocks and therefore did
    # not embed unused source parameters.
    return sampled_file_signature(source_path)


def build_contract(
    config: R7Config, *, encoder_ckpt: str = "", dual_ae_ckpt: str = "",
    r7_ckpt: str = "", include_signatures: bool = True,
) -> Dict[str, Any]:
    """Build a path-independent contract; signatures are optional while training."""
    config.validate()
    contract = {
        "schema": R7_CONTRACT_SCHEMA,
        "config": asdict(config),
        "layout": {
            "rgb": ["B", config.seq_len, 3, config.target_size, config.target_size],
            "dual": ["B", config.seq_len, config.latent_grid,
                     config.latent_grid, config.geo_dim + config.tex_dim],
            "r7": ["B", config.latent_seq_len, config.latent_grid,
                   config.latent_grid, config.latent_dim],
            "anchor_chunks": [1, config.latent_seq_len - 1],
            "channel_split": [config.geo_latent_dim, config.tex_latent_dim],
        },
        "sampling": {
            "frames": config.seq_len,
            "duration_seconds": config.clip_duration_seconds,
        },
        "signatures": {},
    }
    if include_signatures:
        for name, path in (("streamvggt", encoder_ckpt),
                           ("source_dual_ae", dual_ae_ckpt),
                           ("r7", r7_ckpt)):
            if path:
                contract["signatures"][name] = sampled_file_signature(path)
    return contract


def validate_contract(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Require exact structure and any explicitly expected signatures."""
    if actual.get("schema") != R7_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported R7 contract schema {actual.get('schema')!r}")
    actual_config = dict(actual.get("config", {}))
    expected_config = dict(expected.get("config", {}))
    # JSON round-trips tuples as lists; normalize this sole sequence field.
    if "levels" in actual_config:
        actual_config["levels"] = tuple(actual_config["levels"])
    if "levels" in expected_config:
        expected_config["levels"] = tuple(expected_config["levels"])
    if actual_config != expected_config:
        raise ValueError("R7 representation contract mismatch in config")
    for key in ("layout", "sampling"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"R7 representation contract mismatch in {key}")
    actual_sigs = actual.get("signatures", {})
    expected_sigs = expected.get("signatures", {})
    if expected_sigs:
        for name, signature in expected_sigs.items():
            if actual_sigs.get(name) != signature:
                raise ValueError(f"R7 representation signature mismatch: {name}")
    elif actual_sigs:
        # Structural-only validation intentionally ignores sampled signatures.
        return


def validate_resume_contract(
    saved: Mapping[str, Any], current: Mapping[str, Any],
) -> None:
    """Strong same-stage check, allowing a saved accepted-R7 self-signature."""
    validate_contract(saved, current)
    current_sigs = current.get("signatures", {})
    for name, signature in saved.get("signatures", {}).items():
        if name == "r7":
            continue
        if current_sigs.get(name) != signature:
            raise ValueError(f"R7 resume signature mismatch: {name}")


@torch.no_grad()
def encode_dual(encoder, compressor, tex_encoder, frames, encoder_dtype):
    """RGB [B,T,3,H,W] -> dual latents in shared BTHWC layout."""
    tokens, psi = encoder(frames.to(dtype=encoder_dtype))
    stripped = strip_special_tokens(tokens, psi)
    geo = compressor([token.float() for token in stripped])
    geo = geo.permute(0, 1, 3, 4, 2).contiguous().float()
    tex = tex_encoder(frames.float()).float()
    if geo.shape[:4] != tex.shape[:4]:
        raise RuntimeError(f"dual latent shape mismatch: {geo.shape} vs {tex.shape}")
    return geo, tex


def encode_r7(tokenizer, geo, tex):
    return tokenizer.encode(geo, tex)


def decode_r7(tokenizer, latent):
    return tokenizer.decode(latent)


def decode_rgb(tokenizer, decoder, latent, frames_chunk_size=None):
    geo, tex = decode_r7(tokenizer, latent)
    return decoder(geo, tex, frames_chunk_size=frames_chunk_size)
