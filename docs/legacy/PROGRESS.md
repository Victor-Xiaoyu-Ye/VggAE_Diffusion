# VggAE-Diffusion 进度 (2026-06-10)

## 当前状态

### Phase 1: 原始 Autoencoder（无 I_0 条件）

| 实验 | 配置 | 数据 | epoch | PSNR | 状态 |
|------|------|------|-------|------|------|
| exp-1 | v1, base=256, 1RB | 10K | 50 | 16.9 dB | ✅ |
| exp-1-big | v2, base=384, 2RB, pixel | 10K | 120 | **20.6 dB** | ✅ 当前最优 |

结论：decoder 天花板 ~22 dB。瓶颈不在压缩比，在 decoder 架构（无 skip connection，纯上采样）。

### Phase 2: Diffusion（无 I_0 条件）

| 实验 | backbone | time | decoder | 结果 |
|------|----------|------|---------|------|
| exp-1 | DiT 95M | LogitNormal | exp-1 v1 | loss 0.20, 采样噪声 |
| exp-2 | DiT 95M | Uniform | exp-1 v1 | loss 升高, 无改善 |
| exp-3-rescale | DiT 95M | Uniform+rescale | exp-1-big v2 | z 分布匹配, RGB 坍缩 |
| Wan compact | Wan 1.3B | Uniform | exp-1 v1 | gen std=25, mode collapse |

结论：三个 backbone 全部失败。根因不在 backbone，在 decoder 对生成 z 过度敏感——即使 z 分布完美匹配，decoder 仍坍缩。

### Phase 3: I_0 条件解码器 🔄

**核心思想**：VGGT 提供几何，第一帧 I_0 提供外观。Decoder 学会 warp 而非凭空生成颜色。

**架构**：

```
I_0 (第一帧) → AppearanceCNN (0.57M, bf16) → {f36, f72, f144}

z_geo [B,8,18,18,512] (所有帧, fp32)
  → stem (18×18)
  → up0 (18→36) + CrossAttn(f36, 128ch)  ← 全局 warping
  → up1 (36→72) + CrossAttn(f72, 64ch)    ← 精细对齐
  → up2 (72→144)                           ← 无 I_0
  → up3 (144→288)                          ← 遮挡补全
  → up4 (288→576)
  → RGB [B,8,518,518,3]
```

**训练**：跨帧采样 (I_A → appearance, I_B → geometry, 间隔 1-4 帧)
Decoder: base_dim=256, 1 ResBlock, ~30M（缩量版防 OOM）

**当前结果**：epoch 17, loss 0.72→0.36, PSNR ~19 dB（仍在收敛）
跨帧重建正常（t0 和 t7 PSNR 几乎相同），证明 decoder 学会了 warp。

### Phase 4: Diffusion + I_0 解码器（待验证）

Diffusion 只生成 z_geo（几何），解码时用 I_0 做外观参考。
rescale 保留（修复 SNR），Uniform 时间采样。

## 关键发现

### Decoder 实验

1. 压缩比不是瓶颈（16× vs 3×, PSNR 差 <1dB）
2. 大模型 + 小数据 = 更差
3. 4DLangVGGT 用 UNet + skip connections → 15M 就做到 22dB
4. 我们的纯上采样 decoder 缺 skip → 从 18×18 到 518×518 全凭 "画"
5. I_0 条件解码解决了核心矛盾：不需要 decoder "画" 颜色

### Diffusion 实验

1. LogitNormal 时间采样导致 t>0.7 区域 OOD
2. z_g std=0.21 导致 SNR 过低 → rescale ×4.2 修复
3. 三个 backbone 失败 → 不是 backbone 问题，是 decoder 敏感度问题
4. I_0 decoder 可能让已有 diffusion ckpt 直接出图（待验证）

### 时间线

```
V1: Wan + raw token (2048-dim) → mode collapse
V2: from-scratch DiT + compact latent → loss 收敛但采样噪声
V3: Wan + compact latent → 更差
V4: rescale + DiT → z 匹配但 RGB 坍缩
V5: I_0 条件 decoder → 重建 ~19dB, 收敛中
V6: I_0 + diffusion → 待验证
```

## 待做

1. I_0 decoder 继续训练，目标 >22dB PSNR
2. 用 I_0 decoder 验证 diffusion 采样（即使 z_geo 不完美也应出结构）
3. 如果 diffusion 出图 → 正式训练 diffusion
4. 数据扩展到 50K+
5. Ascend 分支同步
