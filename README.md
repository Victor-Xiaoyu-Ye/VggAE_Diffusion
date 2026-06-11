# VggAE Diffusion

Geometry-aware video reconstruction and generation using frozen StreamVGGT
features, a compact generative tokenizer, an I0-conditioned decoder, and latent
flow matching.

This branch targets Ascend 910B. Active training uses `torch_npu`, HCCL, and
FP16 with gradient scaling.

## Active Workflows

Before launching, edit the cluster paths once in:

```text
scripts/spatialvid_config.sh
```

The active scripts automatically create deterministic, non-overlapping
SpatialVID CSV files under `RUN_ROOT/metadata/`. You do not need to extract
overfit or validation CSV files manually. Experiment hyperparameters are
grouped at the top of each `.sh`; only distributed launch settings are read
from environment variables.

### 10K experiments

Use [`scripts/10k/`](scripts/10k/README.md) to reproduce and validate the
current online pipeline:

1. Train the compact geometry autoencoder.
2. Train the I0-conditioned RGB decoder.
3. Overfit the decoder and diffusion model on a tiny set.
4. Train I0-conditioned compact diffusion.
5. Sample and evaluate.

Run all bring-up gates and write one result report with:

```bash
bash scripts/10k/run_validation_suite.sh
bash scripts/10k/collect_results.sh
```

This path decodes video and runs StreamVGGT inside the training loop. It is not
intended for the 10M run.

### Large-scale training

Use [`scripts/scale/`](scripts/scale/README.md). The scale path freezes the
tokenizer, caches compact I0/future-residual latent tar shards, and trains the
generator by optimizer step without video decoding in the diffusion loop.

The design, validation gates, storage estimates, and Wan recommendation are in
[`SCALE_TRAINING.md`](SCALE_TRAINING.md).

The supported/experimental/legacy version matrix is in
[`VERSION_STATUS.md`](VERSION_STATUS.md).

## Current Model Contract

```text
video -> frozen StreamVGGT -> tokenizer -> compact latent
I0 RGB -> appearance CNN ------------------------+
                                                  |
I0 latent + seven generated residual latents -> I0 decoder -> RGB video
```

Frame 0 is observed and is not a diffusion target.

## Legacy Experiments

Older DPT decoder, raw-token diffusion, reduced-Wan, and non-I0 compact scripts
are isolated in [`scripts/legacy/`](scripts/legacy/README.md). They are retained
only for checkpoint reproduction and should not be used for new scale runs.

Historical design notes are in [`docs/legacy/`](docs/legacy/README.md).

## Setup

```bash
# Install the torch/torch_npu/torchvision versions matching the cluster CANN
# release first.
pip install -r requirements.txt
```

Wan source is vendored under `Wan2.1/`; checkpoints are downloaded separately.
