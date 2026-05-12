# VggAE-Diffusion：几何潜在空间中的视频生成

## 一、Motivation

现有视频生成方法（SVD、Wan2.1）在 VAE latent space 中做 diffusion。VAE latent 擅长压缩像素，但**不编码几何结构**。StreamVGGT 是几何基础模型，其 token space 天然编码跨帧的几何对应关系。

**核心假设**：在 StreamVGGT 的几何潜在空间中做 diffusion，生成的视频会有更好的 3D 结构一致性和时序连贯性。

|                 | Latent Space   | 数据        |
|-----------------|---------------|------------|
| SVD/Wan2.1      | VAE (像素压缩)  | 百万视频    |
| GLD             | DA3 (几何)    | 几千多视图  |
| **VggAE-Diffusion** | **StreamVGGT (几何+时空)** | **1万视频** |

## 二、Pipeline

```
数据: mp4 (+ zip/EXR depth) → read_video_frames (8帧, 518×518)

Encoder (StreamVGGT, frozen, bf16):
  [B,8,3,518,518] → 24 alternating-attention blocks
  → tokens_list[24]×[B,8,1+4+1369,2048]
  → strip_special_tokens → tokens_list[24]×[B,8,1369,2048]

Token Processing:
  normalize → diffusion: 1 boundary level (level 11)
            → decoder:  4 DPT levels [4,11,17,23]

Phase 1 — Decoder 训练:
  tokens → ViTDecoder (token-native transformer, 主方案)
        → DPTHead    (卷积多尺度融合, 对比方案, train_decoder_dpt.py)
  Loss: L1 + LPIPS + temporal_diff + image_gradient (+ depth_L1 for DPT)

Phase 2 — Diffusion 训练 (OT-CFM):
  noise → VideoDiT (12层, 768-dim, spatial+temporal attention) → tokens
  + frozen decoder auxiliary loss (RGB-space)
  + 可选 CLIP text conditioning

Inference:
  noise → ODE (Euler/Midpoint, 50步) → tokens → decoder → RGB video
```

## 三、Decoder 对比

|              | ViTDecoder (主线)                    | DPTHead (对比)                    |
|--------------|--------------------------------------|----------------------------------|
| 风格          | Token-native transformer (GLD/MAE)   | 卷积多尺度融合 (MiDaS/DPT)        |
| 参数量         | 16M (dim=512, depth=4)              | 33M                              |
| 跨level融合    | Learned fusion MLP                   | Refinenet 渐进融合 + resize       |
| 训练脚本       | `train_decoder.py`                   | `train_decoder_dpt.py`           |
| 特点           | 更轻量, 与 token space 自然匹配       | 多尺度金字塔, 成熟                 |

ViTDecoder 关键设计：
- 2048 → decoder_dim 投影
- Level embedding + spatial position embedding
- Transformer blocks (跨 patch + 跨 level attention)
- **Learned level fusion**: `Linear(4D→D) + GELU + Linear(D→D)` (非简单平均)
- Zero-init output projection

DPTHead 额外支持：`--output_depth` + `--depth_root` 多任务学习（借鉴 GLD）

## 四、深度数据

格式：zip 包内 EXR 文件，与视频 `group_0001/{id}.mp4` 一一对应

```
depths/SpatialVID/depths/group_0001/{video_id}.zip
  ├── 00000.exr  (1 channel 'Z', float32, 米制深度)
  ├── 00001.exr
  └── ...
```

视频帧和深度帧共享同一组采样索引。深度模式下关闭 temporal jitter 保证对齐。

## 五、项目结构

```
VggAE-Diffusion/
├── data/
│   ├── token_utils.py          # 归一化、level 选择、增强、编解码
│   └── video_dataset.py        # SpatialVidDataset (mp4 + zip/EXR depth)
├── models/
│   ├── video_dit.py            # VideoDiT (spatial + temporal attention)
│   ├── flow_matching.py        # OT-CFM (flow matching + ODE sampling)
│   ├── clip_encoder.py         # Frozen CLIP ViT-L/14
│   └── vit_decoder.py          # ViTDecoder (token-native, learned fusion)
├── streamvggt/                 # StreamVGGT encoder (from Meta/VGGT)
│   ├── models/aggregator.py    # 24 alternating-attention blocks
│   ├── heads/dpt_head.py       # DPTHead (RGB + optional depth)
│   └── layers/                 # Transformer primitives
├── utils/
│   ├── video_io.py             # mp4 读取 + EXR depth + temporal jitter
│   ├── decoder_loader.py       # 共享 decoder 加载 (auto-detect ViT/DPT)
│   ├── training.py             # EMA, AdamW, cosine schedule
│   └── distributed.py          # DDP setup
├── scripts/
│   ├── train_decoder.sh        # ViTDecoder 训练
│   ├── train_decoder_dpt.sh    # DPTHead 训练 (含 depth)
│   ├── train_diffusion.sh      # Diffusion 训练
│   ├── reconstruct.sh          # 重建 (encoder→decoder, PSNR/SSIM)
│   └── sample.sh               # 生成视频
├── train_decoder.py            # Phase 1: ViTDecoder (主线)
├── train_decoder_dpt.py        # Phase 1: DPTHead (对比)
├── train_diffusion.py          # Phase 2: OT-CFM diffusion
├── reconstruct.py              # 重建评估 (兼容 ViT/DPT)
├── sample.py                   # 采样生成 (兼容 ViT/DPT)
├── overfit_single.py           # Debug: 单视频过拟合
├── analyze_levels.py           # Level ablation 工具
├── visualize_latent_space.py   # PCA、距离分析
└── compute_token_stats.py      # Step 0: Welford 统计量
```

## 六、启动命令

```bash
# Step 0: 计算 token 统计量 (如未计算)
python compute_token_stats.py --csv_path ... --video_root ... --out_path ckpts/token_stats.pt

# Phase 1: 训练 decoder
bash scripts/train_decoder.sh       # ViTDecoder (主线)
bash scripts/train_decoder_dpt.sh   # DPTHead + depth (对比)

# Phase 2: 训练 diffusion
bash scripts/train_diffusion.sh

# 评估
python reconstruct.py --video_path ... --encoder_ckpt ... --decoder_ckpt ...
python analyze_levels.py --video_list ... --encoder_ckpt ... --decoder_ckpt ...

# 生成
python sample.py --flow_ckpt ... --decoder_ckpt ... --text_prompt "..."
```

## 七、训练 Trick

| 类别 | Trick | 默认值 |
|------|-------|--------|
| Decoder 鲁棒性 | Level dropout | 0.15 |
| Decoder 鲁棒性 | Boundary-only training | prob=0.25 |
| Decoder 鲁棒性 | Token noise | std=0.02 |
| 数据增强 | Temporal jitter | 随机起始帧+步长抖动 |
| 正则化 | EMA | decay=0.999 (decoder) / 0.9999 (diffusion) |
| 正则化 | Input token noise (diffusion) | std=0.005 |
| 训练稳定 | Zero-init output projection | ViTDecoder + VideoDiT |
| 训练稳定 | bf16 mixed precision | ✓ |
| 训练稳定 | Gradient checkpointing | ✓ |
| 多任务 | Depth head (DPT only) | weight=0.1 |
| 损失 | L1 + LPIPS(0.1) + temporal(0.05) + gradient(0.05) | decoder |
| 辅助 | Frozen decoder auxiliary loss | weight=0.05, every=1 batch |

## 八、下游脚本兼容性

`reconstruct.py` / `sample.py` / `train_diffusion.py` / `overfit_single.py` 均通过 `utils/decoder_loader.py` 自动检测 checkpoint 类型（ViT vs DPT），无需手动指定。

## 九、待做事项

1. 训练 ViTDecoder，评估 PSNR/SSIM，与 DPTHead 对比
2. 训练 DPTHead + depth，验证多任务对 RGB 的提升
3. `analyze_levels.py` 量化 level 4/11/17/23 的信息含量
4. 从零训练 diffusion (100 epoch)，监控 loss + decoder auxiliary
5. Wan2.1 weight init 对照实验 (等 from-scratch 稳定后)
6. Causal temporal attention → 自回归长视频生成
