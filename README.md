# VggAE Diffusion

Active research code for geometry-aware video generation with frozen
StreamVGGT compact latents, an I0-conditioned decoder, and Compact DiT/Wan
latent generators.

For agent handoff and current project state, read:

```text
AGENTS.md
PROJECT_CONTEXT.md
```

## Active Scale Path

The main ModelArts/Ascend workflow is under `scripts/scale/`.

```bash
bash scripts/scale/smoke_wan_compact.sh
bash scripts/scale/05_train_wan_compact.sh
bash scripts/scale/06_sample_wan_compact.sh
```

## Current Model Contract

```text
I0 RGB -> StreamVGGT + tokenizer -> z0
future frames -> StreamVGGT + tokenizer -> z1...z7
generator target = z1...z7 - z0
decoder input = [z0, z0+r1, ..., z0+r7] + I0 appearance
```

Frame 0 is observed and is not a diffusion target.

## Documentation

- `AGENTS.md`: first-read instructions for future agents.
- `PROJECT_CONTEXT.md`: goals, architecture, decisions, conventions, known
  issues, research risks, and next tasks.
- `scripts/h200/README.md`: active reconstruction-first probe plan (H200).
- `scripts/scale/README.md`: runnable scale-stage command order (paused
  until reconstruction passes the gate).
- `TOKEN_STATS.md`: latent normalization contract.
- `SCALE_TRAINING.md`: detailed scale plan and validation gates.

## Setup

```bash
# Install the torch/torch_npu/torchvision versions matching the cluster CANN
# release first.
pip install -r requirements.txt
```

Wan source is vendored under `Wan2.1/`; checkpoints are downloaded separately.
