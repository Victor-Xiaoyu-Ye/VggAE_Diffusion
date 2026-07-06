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
