# VggAE-Diffusion 架构设计 (Compact Latent 版)

## 整体流水线

```
Video x [B,S,3,518,518]
    │
    ▼
StreamVGGT Encoder (frozen, bf16)
    │  24 levels × [B,S,1369,2048]
    ▼
Generative Tokenizer A (13.7M, trainable)
    │  per-level LN → 1×1 Linear(2048→512) → gated fusion
    │  → spatial compress 37→18 → temporal mixer
    ▼
z_g [B,S,18,18,512]  ← compact generative latent
    │                        │
    ▼                        ▼
CompactDecoder G          CompactLatentDiT (Phase 2)
(27.4M, trainable)       (~150M, trainable)
    │                        │ flow matching on z_g
    ▼                        ▼
RGB + Depth               ẑ_g → Decoder G → RGB
[1,S,518,518,4]
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 条件注入 | concat time embedding | V3 验证：adaLN time_var=0.000006 |
| 空间建模 | 18×18 grid, avg_pool | V4：空间 attention 仅 5.6% 改善 |
| 时序建模 | depthwise conv1d (tokenizer) | V7：时序 attention 仅 1.2% 改善 |
| 多层融合 | gated fusion (RAEv2) | V5：简单 mean 归一化 loss 更差 |
| Backbone | from-scratch DiT, no Wan | 不用 adaLN, 不用 pretrained prior |
| Target | single level (11) | V5+V6：multi-level mean→11 PSNR 暴跌 |

## 各模块详设

### Generative Tokenizer A
```
Input: tokens_list[4,11,17,23] × [B,S,1369,2048]
  → PerLevelNorm (LN + learnable affine)
  → 1×1 Linear(2048→512) per level
  → GatedLevelFusion:
      concat all levels → gate_net → softmax weights
      z = Σ gate_i · f_i
  → reshape [B,S,37,37,512]
  → adaptive_avg_pool 37→18
  → TemporalMixer: depthwise Conv1d + gated residual
  → output_refine: 2× Conv2d
Output: z_g [B,S,18,18,512], z_g_flat [B,S,324,512]
```

### CompactDecoder G (v1 / exp-1)
```
Input: z_g [B,S,18,18,512]
  → Stem: 2× ConvBlock + 2× ResBlock @ 18×18
  → Bilinear upsample + ResBlock: 18→36→72→144→288→576
  → TemporalAttn: multihead self-attn across frames @ 36×36
  → Final refine: ConvBlock + ResBlock
  → RGB head: Conv2d(64→3) + Sigmoid
  → Depth head: Conv2d(64→1) + Sigmoid
Output: [B,S,518,518,4] (BHWC)
Decoder params: 27.4M (v1, base_dim=256)
```

### CompactLatentDiT
```
Input: z_flat [B,S,324,512] + time t
  → Sinusoidal time emb (256→768, float32 compute)
  → WideHead: concat token+time → Linear(1280→3072) → Linear(3072→768)
  → spatial_pos [1,1,324,768] + temporal_pos [1,8,1,768]
  → 8× Spatial DiTBlock (within-frame self-attn)
  → cross-attn to CLIP text every 2 blocks
  → 4× Temporal DiTBlock (cross-frame at each position)
  → LayerNorm → WideHead(768→3072→512) zero-init
Output: v [B,S,324,512] predicted velocity
DiT params: ~150M (model_dim=768, spatial=8, temporal=4, heads=12)
```

### 训练配置

| Phase | 训练内容 | 冻结 | Loss | 噪声增强 |
|-------|---------|------|------|---------|
| 1 | Tokenizer + Decoder | Encoder | L1 + LPIPS + grad + temporal + latent_reg | z_g + σ·ε (ramp) |
| 2 | DiT (flow matching) | Encoder + Tokenizer + Decoder | MSE(v_pred, v) + λ·L1(RGB) | — |

## 维度压缩

```
Raw tokens:      8F × 1369 × 2048 = 22M dims/frame
Compact latent:  8F × 324   × 512  = 166K dims/frame
压缩比: 133×
```

## 两版 decoder

| | v1 (exp-1) | v2 (big) |
|--|-----------|---------|
| base_dim | 256 | 384 |
| ResBlocks/stage | 1 | 2 |
| Upsample | bilinear | pixel-shuffle |
| Temporal blocks | 1 (@36×36) | 2 (@36×36, @72×72) |
| Final refine | ConvBlock + ResBlock | 2×(ConvBlock + ResBlock) |
| Params | 27.4M | ~55M |
