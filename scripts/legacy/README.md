# Legacy Experiment Scripts

These scripts reproduce historical experiments. They are not active training
entry points and retain machine-specific paths that must be edited before use.

| Family | Status |
|---|---|
| `train_decoder*` | Historical DPT/ViT decoder comparisons |
| `train_diffusion.sh` | Raw 2048-dim StreamVGGT token diffusion; intractable at scale |
| `train_diffusion_wan*.sh` | Historical Wan LoRA/raw-token experiments |
| `train_wan_compact_diffusion.sh` | Compact Wan experiment without the current cached I0-residual contract |
| `train_compact_diffusion.sh` | Non-I0 compact baseline |
| `train_autoencoder.sh` | Original v1 compact decoder baseline |
| `sample.sh`, `reconstruct.sh` | Sampling for historical raw-token checkpoints |

Known limitations:

- The old Wan text path used CLIP rather than Wan's native UMT5 representation.
- Raw-token diffusion models tens of millions of dimensions per clip.
- Several scripts contain checkpoint-specific `--resume` paths.
- These workflows run StreamVGGT online and are not suitable for 10M videos.

The Python modules remain available so existing checkpoints can still be
inspected. New training should use `scripts/10k/` or `scripts/scale/`.
