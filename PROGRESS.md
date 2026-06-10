# VggAE-Diffusion 进度

## 当前状态 (2026-06-02)

### Phase 1: Autoencoder

| 实验 | 配置 | 数据 | epoch | PSNR | 状态 |
|------|------|------|-------|------|------|
| exp-1 | v1, base=256, 1RB, bilinear | 10K | 50 | 16.9 dB | ✅ |
| exp-1-big | v2, base=384, 2RB, pixel-shuffle | 10K | 120 | **20.6 dB** | ✅ 当前最优 |

### Phase 2: Diffusion

| 实验 | backbone | time sampling | decoder | 状态 |
|------|----------|--------------|---------|------|
| exp-1 (compact) | from-scratch DiT 150M | LogitNormal | exp-1 v1 | ✅ loss 0.20, 采样噪声 |
| exp-2 (uniform-t) | from-scratch DiT 150M | Uniform | exp-1 v1 | ❌ loss 升高, 无变化 |
| exp-1 (Wan) | Wan 1.3B + 双时间条件 | Uniform | exp-1-big v2 | 🔄 训练中 |

### Phase 3: 纹理增强

- 设计完成，待实现
- TextureEncoder (3M) 从 RGB 提取高频/颜色 → z_tex
- TexturePredictor 从 z_geo 预测 z_tex（推理时用）
- 训练时 concat(z_geo, z_tex) 送入 decoder

---

## 关键发现

### Decoder 实验总结

1. **StreamVGGT 特征信息足够**——旧 DPTHead 全量 22dB，133× 压缩后 20.6dB，只丢 1-2dB
2. **压缩比不是瓶颈**——16× (18.1dB) vs 3× (18.8dB)，差 <1dB
3. **大模型 + 小数据 = 更差**——relaxed_big (121M) 14.4dB < tight (27M) 18.1dB
4. **200视频/30epoch 天花板 ~20dB**，10K/120epoch ~20.6dB
5. **StreamVGGT >> DINO** (19.7 vs 16.8 dB) —— 2048-dim 信息密度远超 768-dim

### Diffusion 实验总结

1. **LogitNormal 时间采样导致 t>0.7 区域 OOD**——模型在 t=1 时 cos_sim=-0.12
2. **改 Uniform 采样后 loss 反而升高**——说明问题不只是时间分布
3. **from-scratch DiT concat conditioning 在深网络中时间信号衰减**
4. **Wan + 双时间条件（concat input + adaLN blocks）** 正在验证中

### 架构演进

```
V1: Wan + raw VGGT token (2048-dim × 37×37) → mode collapse
V2: from-scratch DiT + compact latent (512-dim × 18×18) → loss 收敛但采样噪声
V3: Wan + compact latent + 双时间条件 → 训练中
V4 (planned): Wan + compact latent + 纹理补偿 → 待实现
```

---

## 待解决问题

1. Diffusion 采样噪声 —— Wan compact 版本能否解决？
2. Decoder PSNR 天花板 ~22dB —— 纹理编码器能否突破？
3. 数据量 —— 10K 可能不够，需要 50K+
4. Ascend 分支 —— 代码已移植，待跑
