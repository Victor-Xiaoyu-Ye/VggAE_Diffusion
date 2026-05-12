# VggAE-Diffusion: Video Generation in Geometric Latent Space

Diffusion-based video generation operating in StreamVGGT's geometrically-aware token space, rather than traditional VAE latent space.

## Pipeline

```
Video → StreamVGGT (frozen) → Tokens [B,8,1369,2048]
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                                           ▼
        Decoder Training                          Diffusion Training
   tokens → RGB reconstruction               noise → Wan2.1/ViTDiT → tokens
   (DPTHead / ViTDecoder / DPTUNet)          (OT-CFM flow matching)
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    ▼
                            Generated Video
```

## Project Structure

```
├── data/              # Dataset + token utilities
├── models/            # ViTDecoder, ViTDiT, WanAdapter, DPTUNet
├── streamvggt/        # StreamVGGT encoder + DPTHead
├── utils/             # Video IO, training helpers, decoder loader
├── Wan2.1/            # Wan2.1 source (submodule / copy)
├── scripts/           # Training shell scripts
├── train_decoder*.py  # Decoder training entry points
├── train_diffusion*.py # Diffusion training entry points
├── sample.py          # Video generation
├── reconstruct.py     # Reconstruction evaluation
└── compute_token_stats.py  # Token normalization stats
```

## Setup

```bash
pip install -r requirements.txt

# Download required HuggingFace models (see HF_MODELS.md)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1/checkpoints/Wan2.1-T2V-1.3B
```

## Quick Start

```bash
# 1. Compute token statistics
python compute_token_stats.py --csv_path ... --video_root ... --out_path ckpts/token_stats.pt

# 2. Train decoder (choose one)
bash scripts/train_decoder_dpt_unet.sh   # DPT + UNet skip connections (recommended)
bash scripts/train_decoder_dpt.sh        # DPTHead (refinenet fusion)
bash scripts/train_decoder.sh            # ViTDecoder (GLD-style)

# 3. Train diffusion (choose one)
bash scripts/train_diffusion_wan.sh      # Wan2.1 LoRA adapter (recommended)
bash scripts/train_diffusion.sh          # From-scratch VideoDiT

# 4. Generate samples
bash scripts/sample.sh
```

## Decoder Architectures

|              | DPTUNet | DPTHead | ViTDecoder |
|--------------|---------|---------|------------|
| Type         | DPT + UNet skip | Refinenet fusion | Transformer |
| Params       | 11.5M   | 33M     | 262M       |
| Key feature  | Concat skip | Additive residual | Channel-concat |

## Diffusion Architectures

|              | Wan2.1 LoRA | From-scratch VideoDiT |
|--------------|-------------|----------------------|
| Backbone     | Wan2.1 1.3B pretrained | 12-layer DiT |
| Trainable    | 43M (LoRA + adapters) | 300M |
| Text cond    | ✓ (CLIP) | ✗ (optional) |

## References

- [4DLangVGGT](https://arxiv.org/abs/2512.05060) - RGB reconstruction from VGGT features
- [GLD](https://arxiv.org/abs/2603.22275) - Geometric Latent Diffusion
- [LaSt-ViT](https://arxiv.org/abs/2602.22394) - ViT feature analysis
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Video diffusion backbone
