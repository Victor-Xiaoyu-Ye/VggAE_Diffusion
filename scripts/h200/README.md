# H200 Reconstruction-First Experiments

This directory is the **4-card H200** experiment path for the
reconstruction-first phase. It is distinct from both the ModelArts Ascend
scale path (`scripts/scale/`) and the local A100 10k path
(`scripts/10k/`).

## Why this path exists

The compact latent currently reconstructs at ~20 PSNR with grid textures,
which caps every generation experiment. Before any further scale diffusion
runs, we must answer: **does the frozen StreamVGGT feature space carry
enough information (especially RGB high-frequency) to reconstruct video at
high PSNR?** The answer decides whether to build a two-stream latent
(`z_geo` + `z_app` from shallow VGGT levels) or keep a single geometry
latent and let the decoder hallucinate RGB high-frequency via perceptual +
adversarial training.

## Environment

`h200_env.sh` sets the dataset, encoder checkpoint, output root, and
4-GPU torchrun launcher. Source it before any launcher:

```bash
source scripts/h200/h200_env.sh
```

Defaults (override by exporting before sourcing):

```text
VGGAE_DATASET_ROOT=/home/yexiaoyu/data/spatial-vid-hq-oft
VGGAE_ENCODER_CKPT=/home/yexiaoyu/data/StreamVGGT/checkpoints.pth
VGGAE_H200_RUN_ROOT=$PROJECT/outputs/h200
VGGAE_NUM_GPUS=4
```

## Probe sequence

Run in this order. Each writes its own output directory under
`$VGGAE_H200_RUN_ROOT/probes/`.

### E1 — raw feature RGB reconstruction ceiling

Concat all four DPT levels at the native 37x37 grid, project to 512-dim,
decode with a strong resize-conv decoder. **No tokenizer compression.**
This is the RGB reconstruction ceiling of the frozen feature space.

```bash
bash scripts/h200/probe_e1_raw_recon.sh
```

Decision rule:
- PSNR >= 28  -> feature space carries RGB high-freq; two-stream latent
  is worth building.
- PSNR <= 23  -> features do not carry RGB high-freq by construction
  (VGGT is trained on geometry, not appearance); use single geometry
  latent + decoder-side hallucination.

### E2 — per-level reconstruction

One VGGT level at a time, same decoder. Identifies which level carries the
most reconstructable information and tests the "deep LayerNorm flattens
high-frequency" hypothesis.

```bash
bash scripts/h200/probe_e2_per_level.sh
```

Runs level 4, 11, 17, 23 sequentially. Compare per-level PSNR.

### E3 — grid artifact isolation

Same compressed pipeline, three decoder/upsample variants:
- (a) PixelShuffle (current default)
- (b) resize-conv (bilinear + conv)
- (c) aligned spatial compress (crop 36 -> stride-2 conv -> 18)

```bash
bash scripts/h200/probe_e3_grid_isolation.sh
```

Identifies which component produces the grid texture.

## E5 run matrix (2026-07-14, 4x H200)

The single most likely fix (survey-driven, see PROJECT_CONTEXT.md
"2026-07-14 survey additions"): avg-pool packing low-passes the high
frequency z_tex exists to carry, and z_tex regularized toward N(0,1)
drifts off z_geo's distribution family. Run in this order — each run is
~1 day on 4 cards; stop early when the decision is already forced.

```bash
# R0 — smoke (~10 min, 1 card): exercises s2d + match_geo + feat loss in one
#      short run, prints the [encoder] key-match line (Next Tasks #0), and
#      PROBE_SUFFIX keeps its checkpoint out of the real runs' auto-resume.
VGGAE_NUM_GPUS=1 VGGAE_GPU_IDS=0 EPOCHS=1 EVAL_CLIPS=4 PROBE_SUFFIX=smoke \
  TEX_PACK=s2d TEX_REG_MODE=match_geo FEAT_WEIGHT=0.5 \
  bash scripts/h200/probe_e5_texture_recon.sh

# R1 — new best guess: s2d packing + SVG-style stat alignment
TEX_PACK=s2d TEX_REG_MODE=match_geo bash scripts/h200/probe_e5_texture_recon.sh

# R2 — packing ablation control (only if R1 passes): isolate s2d's share
TEX_REG_MODE=match_geo bash scripts/h200/probe_e5_texture_recon.sh

# R3 — geo-only floor (cheap; run after R1, or on a spare card)
TEX_MODE=zero bash scripts/h200/probe_e5_texture_recon.sh

# R4 — anti-bypass proof (required for the paper once R1 passes)
TEX_MODE=tex_only TEX_PACK=s2d bash scripts/h200/probe_e5_texture_recon.sh

# R5 — feature-consistency loss arm (MIRA P-DINO analogue; run if R1
#      lands in the 23-25 gray zone — it lifts fidelity without GAN)
TEX_PACK=s2d TEX_REG_MODE=match_geo FEAT_WEIGHT=0.5 \
  bash scripts/h200/probe_e5_texture_recon.sh
```

Output dirs are derived from the knobs: R1 ->
`probes/e5_oracle_s2d_match_geo`, R2 -> `probes/e5_oracle_avgpool_match_geo`,
R3 -> `probes/e5_zero_avgpool`, R4 -> `probes/e5_tex_only_s2d`, R5 ->
`probes/e5_oracle_s2d_match_geo_feat0.5`. Each auto-resumes from its own
`checkpoint_latest.pt`, so re-running a command continues that arm.

Decision gates (BOTH must pass before any cache rebuild):

1. R1 oracle PSNR >= 25 (32-clip mean, no grid texture in samples).
2. Diffusability spectra sane — run on the R1 checkpoint:

```bash
"$VGGAE_PYTHON_BIN" "$VGGAE_PROJECT/diagnose_latent_diffusability.py" \
  --ckpt  "$VGGAE_H200_RUN_ROOT/probes/e5_oracle_s2d_match_geo/checkpoint_latest.pt" \
  --csv "$VGGAE_EVAL_CSV" --video_root "$VGGAE_VIDEO_ROOT" \
  --encoder_ckpt "$VGGAE_ENCODER_CKPT" \
  --output_json "$VGGAE_H200_RUN_ROOT/probes/e5_oracle_s2d_match_geo/diffusability.json"
```

   Gate: z_tex spatial_band_high not >> z_geo's, top32_var high (few-mode
   eigenspectrum), kurtosis near 3, and the residual streams
   (z_*_residual — what diffusion actually predicts) at least as tame as
   the absolute streams. A latent that passes PSNR but fails this is
   MIRA's "sharper but undiffusable" failure mode — fix packing/reg
   before raising capacity.

3. R4 tex_only must be clearly below R1 oracle (geometry load-bearing).

If R1 still lands ~20: capacity is next (TEX_DIM=384 TEX_BASE_CH=96),
NOT more packing changes — s2d already removed the information bottleneck,
so a persistent plateau means decoder or fusion capacity.

## After the probes

Decision tree (see PROJECT_CONTEXT.md `Research Risks`):

```
E1 PSNR
  >= 28 + E2 shows shallow levels better  -> two-stream latent (z_geo + z_app)
  >= 28 + E2 roughly equal across levels  -> single stream + capacity increase
  <= 23                                    -> single geometry latent + decoder
                                              hallucinates RGB high-freq via
                                              perceptual + adversarial training
```

Only after the reconstruction reaches PSNR >= 25 with no grid texture do
we retrain the tokenizer and rebuild any latent cache.
