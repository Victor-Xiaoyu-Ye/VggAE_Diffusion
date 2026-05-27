# VggAE-Diffusion：几何潜在空间中的视频生成

## 一、核心假设

在 StreamVGGT 的几何 token space 中做 diffusion，利用预训练 encoder 的 3D 几何先验，生成具有更好结构一致性的视频。

|                 | Latent Space   | 数据        | Latent 维度 |
|-----------------|---------------|------------|-------------|
| SVD/Wan2.1      | VAE (像素压缩)  | 百万视频    | ~100K       |
| GLD             | DA3 (几何)    | 几千多视图  | ~770K       |
| RAEv2           | DINOv3 (语义) | 100万图像   | ~262K       |
| **VggAE-Diffusion** | **StreamVGGT (几何+时空)** | **1万视频** | **22M → 660K (降维后)** |

## 二、项目架构

```
视频 [B,8,3,518,518]
  │
  ▼
StreamVGGT Encoder (frozen, 24 blocks)
  │
  ▼
tokens_list[24] × [B,8,1369,2048]
  │
  ├──► Decoder (DPTHead) ──► RGB video [B,8,518,518,3]
  │    Loss: L1 + LPIPS + temporal + gradient + depth
  │
  └──► Diffusion (Wan2.1 DiT + adapter)
       [B,8,1369,2048] → multi-layer mean → Linear 2048→256
       → spatial pool 37→18 → Wan backbone → unpool 18→37
       → Linear 256→2048 → [B,8,1369,2048]
       Loss: flow matching MSE + decoder auxiliary (RGB-space)
```

## 三、Decoder 研究进展

### 实验结果汇总

| 实验 | 架构 | PSNR | LPIPS | 结论 |
|------|------|------|-------|------|
| exp-5-dpt | DPTHead baseline (256f) | 20.9 dB | 0.256 | **最佳 baseline** |
| exp-6-big | DPTHead big (512f) | 22.3 dB | 0.253 | +1.4dB PSNR，LPIPS 几乎不动 |
| exp-7-gan | DPTHead + GAN | 22.3 dB | 0.255 | GAN 无增益，LPIPS 天花板 ~0.25 |
| exp-4-gld | ViTDecoder GLD | ~18 dB | 0.52 | 不如 conv decoder |
| exp-5-big | ViTDecoder big (262M) | ~18 dB | 0.39 | 容量不解决问题 |
| exp-8-mean | DPTHead + all-level mean | 18.2 dB | 0.35 | **变差**——mean 替换全部 level 破坏多尺度 |
| exp-10-mean-bdry | DPTHead + mean→boundary only | 训练中 | - | mean 只替代 level 11，保留多尺度多样性 |

### 核心发现

1. **Decoder 天花板在 ~22dB PSNR，LPIPS 卡在 0.25**。三种架构（DPTHead、ViTDecoder、DPTUNet）、容量翻倍、GAN 均无法突破。瓶颈在 encoder token 本身的 RGB 信息承载能力，不在 decoder 架构。
2. **DPTHead refinenet 是最佳 decoder**——33M 参数碾压 262M ViTDecoder。多尺度金字塔对该任务有强先验。
3. **Multi-layer mean 不适用于 DPTHead**——替换全部 4 level 破坏了多尺度金字塔。boundary-only 模式（mean 替代 level 11，其余保持原始）正在验证。
4. **4DLangVGGT 用极简 UNet 在 HyperNeRF 静态场景获得 26.5dB**——对比说明视频 token 的时序压缩稀释了单帧信息。

### 参考论文

- **RAEv2** (arXiv:2605.18324): Multi-layer mean 降噪 + REPA 加速扩散收敛。用于 ViT flat decoder，不适用于 DPTHead 多尺度架构。
- **GLD** (arXiv:2603.22275): Channel-concat multi-level features + ViT decoder。在 DA3 静态场景实现 35.4dB——多视图比视频简单得多。
- **4DLangVGGT** (arXiv:2512.05060): UNet skip decoder on VGGT tokens。静态场景 26.5dB，验证了 token 信息承载上限。
- **LaSt-ViT** (CVPR 2026): ViT 深层特征"线性化"，有效秩远低于名义维度。解释了深层 level (17/23) 可能信息量低。

## 四、Diffusion 研究进展

### 实验汇总

| 实验 | 架构 | Flow Loss | Epoch | GPU | 结论 |
|------|------|-----------|-------|-----|------|
| exp-diff-1 | VideoDiT from-scratch (300M) | 1.47 | 100 | A100 | 不收敛 |
| exp-2-wo-lora | Wan 1.45B full adapters (无LoRA) | 1.45 | 44 | 910B | 不收敛 |
| exp-3-text | Wan + LoRA + text (GPU) | 1.13 | 49 | A100 | 收敛慢但有效 |
| exp-3-text-ascend | Wan + LoRA + text (NPU) | 1.57 | 68 | 910B | NPU 不稳定 |
| **exp-4-reduced** | **Wan + LoRA + 降维 (34×)** | **1.21** | **50** | **A100** | **Latent 660K → 每步下降更快** |

### 降维方案 (exp-4-reduced)

```
Input:  [B, 8, 1369, 2048]
  ├─ multi-layer mean (levels 4+11+17+23)/4  → SNR 提升
  ├─ Linear(2048→256)                         → channel 降维
  ├─ adaptive_avg_pool(37→18)                 → spatial 降维
  │
  ▼ Wan backbone: [B, 8, 324, 1536]  (660K dims)
  │
  ├─ nearest unpool(18→37)                    → spatial 恢复
  └─ Linear(256→2048)                         → channel 恢复
Output: [B, 8, 1369, 2048]                    → 外部格式不变，decoder auxiliary 兼容
```

**效果**：Latent 从 22M dims → 660K dims（34× 缩减），RAEv2 量级。同 steps 内下降更快，epoch 效率待优化。

### 核心发现

1. **维度灾难是最大瓶颈**：22M-dim continuous flow matching 在 10K 数据上几乎不可训。降维 34× 后每步下降更快。
2. **NPU 910B 训练不稳定**：多个算子（interpolate、sort、linspace）有兼容性问题，需逐个绕过。GPU 版本更稳定。
3. **Decoder auxiliary 在 NPU 上不可用**：DPTHead 的 bilinear interpolate 触发 aclnnSort 崩溃。GPU 上正常工作。
4. **CubiD 离散扩散是下一步方向**：256-dim token 用 dimension-level 离散 mask/predict 替代 continuous flow matching，在高维更稳定。

### 参考论文

- **CubiD** (arXiv:2603.19232): 高维 token 的离散扩散——把生成分解为逐维度 mask/predict，T 步固定。适合 256-dim 降维后的 latent。
- **RAEv2 REPA**: 在 DiT 中间层抽特征匹配 encoder 输出，作为辅助 loss。10× 加速收敛，可用于我们的 adapter。

## 五、关键设计决策

### Level 选取策略

| 方案 | 效果 | 说明 |
|------|------|------|
| 单一 level 11 | **baseline (22.3 dB)** | 当前选择 |
| 4 level concat | DPTHead 标准输入 | 不同分辨率特征互补 |
| mean → all levels | **变差 (18.2 dB)** | 破坏多尺度 |
| mean → boundary only | 待验证 | exp-10 进行中 |

### 降维策略

| 维度 | 当前 | 降维后 | 方式 |
|------|------|--------|------|
| Channel | 2048 | 256 | Learned Linear projection |
| Spatial | 1369 patches | 324 patches | adaptive_avg_pool 2× |
| Temporal | 8 frames | 8 frames | 保持不变 |
| 总维度 | 22M | 660K | **34× reduction** |

## 六、当前实验矩阵

### Decoder (GPU)

| 实验 | 目录 | 状态 |
|------|------|------|
| exp-5-dpt (baseline) | `decoder_dpt/exp-5-dpt` | 完成, PSNR 20.9 |
| exp-6-big (512f) | `decoder_dpt/exp-6-big` | 完成, PSNR 22.3 |
| exp-7-gan | `decoder_dpt/exp-7-gan` | 完成, PSNR 22.3 |
| exp-8-mean (全替换) | `decoder_dpt/exp-8-mean` | 完成, PSNR 18.2 |
| exp-10-mean-bdry | `decoder_dpt/exp-10-mean-bdry` | 训练中 |

### Diffusion

| 实验 | 目录 | 平台 | 状态 |
|------|------|------|------|
| exp-diff-1 (VideoDiT) | `diffusion_level11_dpt` | GPU | 完成, loss 1.47 |
| exp-3-text (Wan LoRA) | `diffusion_wan/exp-3-text` | GPU | 完成, loss 1.13 |
| exp-4-reduced (降维) | `diffusion_wan/exp-4-reduced` | GPU | 完成, loss 1.21 |

## 七、新增模型文件

```
models/
├── video_dit.py              # 12层 from-scratch DiT
├── flow_matching.py          # OT-CFM flow matching
├── wan_adapter.py            # Wan2.1 1.3B adapter (LoRA + i/o proj)
├── wan_adapter_reduced.py    # 降维版 Wan adapter (34× reduction)
├── vit_decoder.py            # ViTDecoder (GLD-style channel-concat)
├── dino_disc.py              # DINO discriminator (GAN)
└── clip_encoder.py           # CLIP ViT-L/14 text encoder
```

## 八、待做事项

### 短期
- [ ] exp-10-mean-bdry 完成，评估 PSNR vs baseline
- [ ] exp-4-reduced 续训到更多 steps，看 flow loss 能否到 <1.0
- [ ] exp-4-reduced 采样一版视频看质量

### 中期
- [ ] CubiD 离散扩散方案（256-dim × 8 groups）
- [ ] REPA-style auxiliary loss（DiT 中间层匹配 encoder feature）
- [ ] 数据扩展到 100K 视频

### 长期
- [ ] Causal temporal attention → 自回归长视频
- [ ] 在降维 latent 上 from-scratch 训练 DiT（对比 Wan transfer）
