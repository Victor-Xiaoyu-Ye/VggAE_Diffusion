# VggAE-Diffusion 重构进度

## 背景

原始方案在 StreamVGGT 的 raw token 空间（2048-dim × 1369 patches × 8 frames = 22M dims）直接做 flow matching diffusion。经过多个实验（exp-2/3/4）发现 Wan+LoRA 和 Wan full fine-tune 均无法收敛，生成样本出现 mode collapse。

## 验证实验结论（verify_diffusion.py, 8 个实验）

| 实验 | 发现 | 架构含义 |
|------|------|---------|
| V1 | flow matching 在 token 空间可行（-45% loss） | 范式本身没问题 |
| V2 | concat/sinusoidal time emb 正常 | 时间条件用 concat |
| V3 | **adaLN 导致 time 响应消失**（time_var=0.000006） | 不用 adaLN |
| V4 | 空间交互仅带来 5.6% 改善 | 空间 attention 非关键 |
| V5 | single level 归一化 loss 优于 multi-level mean | 单 level 作 target |
| V6 | decoder 对 token 噪声鲁棒；旧 decoder 不能直接用 | 需要新 decoder |
| V7 | 时序 attention 帮助极小（1.2%） | 时序交互非关键 |
| V8 | loss 随帧数线性增长，S=8 是合理折中 | 帧率暂不调整 |

**核心发现**：Wan backbone 的 adaLN conditioning 无法在 token 空间学习时间依赖行为，导致模型忽略时间输入，预测时间无关的速度场，引发 mode collapse。同时，token 空间几乎是 per-token 独立的（spatial/temporal attention 帮助很小），不需要 Wan 级别的重 backbone。

## 新架构：三阶段

```
Frozen StreamVGGT Encoder
    ↓
f = {f4, f11, f17, f23}  (可选多层)
    ↓
═══════════════════════════════
  Generative Tokenizer A      ← Phase 1 训练
═══════════════════════════════
  per-level LayerNorm
  → 1×1 Linear 2048→512
  → gated multi-layer fusion
  → spatial compress 37→18
  → light temporal mixer
  → z_g: [B, S, 18, 18, 512]
═══════════════════════════════
    ↓                    ↓
  Diffusion            Decoder G        ← Phase 1 联合训练
  (Phase 2)            DPT-style upsample
  CompactLatentDiT     + temporal attn
  concat time cond     + pixel shuffle
  factorized attn      + RGB/depth heads
    ↓                    ↓
  ẑ_g  ──────────────→ RGB video
```

## 已完成模块

### models/generative_tokenizer.py
- `PerLevelNorm`: per-level LN + learnable affine
- `GatedLevelFusion`: 多层 gated fusion（替代简单 mean）
- `SpatialCompressor`: adaptive_avg_pool 37→18 + pre-pool conv
- `TemporalMixer`: depthwise conv1d + gated residual
- `GenerativeTokenizer`: 主模块，输入 24-level token list，输出 z_g

### models/compact_decoder.py (高容量版)
- base_dim=384, 2×ResBlock per stage
- PixelShuffle 上采样（替代 bilinear）
- 两级 temporal attention（36×36 + 72×72）
- 更深的 final refine
- RGB + depth 双 head

### models/compact_dit.py
- `WideHead`: RAE-style wide input/output projection
- `DiTBlock`: pre-norm transformer block
- `CompactLatentDiT`: factorized spatial-temporal DiT
  - concat time conditioning（非 adaLN）
  - 可选的 text cross-attention
  - zero-init output

### train_autoencoder.py (Phase 1)
- 联合训练 Tokenizer A + Decoder G
- 冻结 StreamVGGT encoder
- 噪声增强：z_g + σ*ε，σ 线性 ramp
- Loss: L1 + LPIPS + gradient + temporal + latent_reg
- 支持 DDP、EMA、gradient checkpointing

### train_compact_diffusion.py (Phase 2)
- 冻结 Tokenizer A + Decoder G
- OT-CFM flow matching on z_g
- CompactLatentDiT backbone
- Decoder 辅助 loss（RGB-space feedback）

### verify_diffusion.py
- 8 个诊断实验，验证 pipeline 各环节

## 训练流程

### Phase 1: 训练 autoencoder
```bash
# 高容量版（推荐，4 卡）
bash scripts/train_autoencoder_big.sh

# 快速验证版（8 卡，base_dim=256）
bash scripts/train_autoencoder.sh
```

目标：
- RGB 重建 PSNR ≥ 旧 decoder 水平（~19-22dB）
- z_g 分布接近 N(0,1)
- decoder 对噪声鲁棒

### Phase 2: 训练 diffusion
```bash
bash scripts/train_compact_diffusion.sh
```
需要 Phase 1 的 checkpoint。

## 关键设计决策

1. **不用 Wan**：adaLN 在 token 空间时间响应消失（V3 验证），改 concat conditioning
2. **不用 spatial-temporal heavy model**：V4/V7 证明 token 空间几乎是 per-token 独立的
3. **单 level target**：V5 证明 single level 归一化 loss 优于 multi-level mean
4. **compact latent**：2048-dim×1369→512-dim×324（133×压缩），降低 diffusion 难度
5. **旧 decoder 不能复用**：输入格式完全不同（24-level×37×37×2048 vs 1×18×18×512）

## 待解决

- LPIPS 需要安装：`pip install lpips`
- Phase 1 训练完成后评估 PSNR，决定是否需要调整 latent 容量
- Phase 2 diffusion 采样质量评估
- 大规模数据扩展（当前 10K → 后续 100K+）
