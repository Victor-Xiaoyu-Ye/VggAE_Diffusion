# R7 首帧条件视频质量实验方案

更新时间：2026-08-17

## 结论先行

`r7_t2_c192_v3` 的失败不能归因于“Wan 还没有训练够久”。当前结果显示：

- R7 t2/c192 codec 的 clean PSNR 约 23.9、LPIPS 约 0.132，但 geometry-motion cosine 约 0.832，未达到生产门槛。
- `decoder_robust` 在 future latent 扰动 `sigma=0.22` 下有效，但只修复 decoder 流形鲁棒性，不会修复 tokenizer 或生成器运动。
- Wan 1.3B 的 teacher-forced x0 MSE 到 14k 仍下降，但 sampled RGB/composite 在约 7k 最好，之后变差。
- chunk 3/4 的 motion cosine 和 expanded geometry motion 接近塌缩。因此不能只看总 x0 MSE，也不能盲目把同一配置续到 30k。

旧实验输出必须保留。所有新表示、cache、stats、diffusion checkpoint 使用新 namespace，禁止覆盖 `r7_t2_c192_v3` 和 `r7_wan13b_t2v_ctx1_fut4_v1`。

## 用户确认的任务和资源

- 任务：首帧条件视频（I2V），输入真实首帧 latent 与 caption，生成后续短视频。
- 训练数据：先用当前 10k subset 做诊断；通过后扩展到完整 SpatialVID-HQ。
- 生成器：只能使用 Wan2.1-T2V-1.3B；本轮不依赖 I2V-14B。
- 允许冻结 Wan VAE 作为训练期教师/对齐目标；最终推理不能依赖 Wan VAE，RGB 仍由 StreamVGGT/R7 decoder 输出。
- 质量门槛：主体/场景稳定，后半段有合理运动和结构，无彩噪、融化、复制、静止塌缩；使用量化 hard gates 和至少 4 个 seed 验收。

## 已完成的基础设施修复

Wan 训练 eval 现在应当在 `samples/` 中生成：

- `stepXXXX_evalYY_ema_grid.png`：`ANCHOR / AE_TARGET / GENERATED` 三行；
- 每个视频的 frame PNG；
- 每个视频的 MP4（若运行环境缺少 ffmpeg/codec，PNG 仍保留，训练不失败）；
- `*_preview.json`：step、video id、seed、CFG、帧数、范围和 MP4 错误信息。

实现位于 `utils/video_preview.py`，并接入 `train_causal_wan_video_diffusion.py`、`sample_causal_wan_video_diffusion.py`、`sample_causal_video_diffusion.py`。默认训练只保存轻量 latent metadata 和 PNG/MP4，完整 float RGB sample pack 需要显式 `--save_debug_sample_pack`。

注意：当前 latent cache 只保存首帧 `i0_rgb`，不保存 future 原始 RGB。因此训练内 `AE_TARGET` 是 R7 autoencoder reconstruction，不是原始视频。raw-RGB 评测必须离线根据 `video_id + window_index` 重新读视频，不能将 AE target 伪称 raw target。

## Phase 0：基线回放和评测修复

1. 用新可视化回放旧 Wan checkpoint 的 step 7000 EMA、step 14000 EMA 和 decoder_robust checkpoint。
2. 固定相同 eval manifest、seed、sample steps、CFG，至少保存 4 个 seeds。
3. 同时报告：
   - generation vs AE target：latent MSE、RGB PSNR/LPIPS；
   - 每个 future chunk 的 motion ratio/cosine；
   - expanded geometry motion ratio/cosine；
   - `ANCHOR / AE_TARGET / GENERATED` PNG/MP4。
4. raw-RGB suite 离线重读 64 个 eval clips，报告 `AE_TARGET vs RAW` 的 codec ceiling，以及 `GENERATED vs RAW` 的真实生成质量。

Wan wrapper 的默认 max steps 已从 30k 收紧到 16k；这不是声称 16k 收敛，而是避免没有 quality evidence 的盲跑。继续训练必须用同一 namespace 的完整 resume，并以 sampled composite 与 late-horizon guards 为准。

## Phase 1：10k 表示诊断矩阵

每个 arm 先做 4k--8k steps，48 卡，eval 每 500 step，PNG/MP4 每次 eval，64 eval clips，4 seeds。任何 arm 连续 3 次 late-horizon guard 变差即停止。

### A. 当前 t2/c192 baseline

复现现有配置，只用于验证新评测与可视化。不能把它晋升为生产方案。

### B. factor-1 codec ceiling

新建独立 representation/cache namespace：不做 temporal fold，每个 RGB frame 保留一个 latent，首帧 + 8 future。先只测 codec round-trip：

- PSNR >= 24.5；
- LPIPS <= 0.12；
- 每帧 geometry-motion cosine >= 0.95；
- 后半段没有明显 motion loss。

factor-1 不是未经测量就认定的最终方案。它的 Wan sequence token 数约为 t2 的 1.8 倍，full temporal attention 的计算/显存约显著增加；若不可承受，使用短窗 overlap rollout，而不是再做有损平均折叠。

### C. t2 + 显式 motion/base 分解

若 factor-1 的质量收益不足，则保留 t2 的 token 预算，但将 target 分成：

- base/content：绝对或相对 anchor 的低频内容；
- motion residual：连续 chunk 差分；
- texture residual：decoder robust 后的高频部分。

模型同时预测 base 和 motion，chunk 3/4 使用递增权重；loss 包含方向 cosine、幅度 ratio、acceleration 和 expanded geo motion。最终再合成 absolute R7 latent。

## Phase 2：Wan 1.3B 正确对齐

当前 `WanCompactAdapter` 虽加载了 Wan trunk，却通过随机 `input_proj/output_proj` 绕过 Wan 原生 VAE/patch language。不能把“加载 checkpoint”当成“保留了 Wan 运动先验”。

### Teacher bridge probe

允许离线使用冻结 Wan VAE：

1. 对同一 9-frame clip 计算 native Wan VAE latent；
2. 提取 native patch embedding 或前若干 Wan block hidden；
3. 预计算 256 clips/约 1k videos 的 teacher cache；
4. 训练 R7 -> Wan hidden/token bridge，加入均值/方差、协方差和 hidden cosine/CKA 对齐；
5. 训练 hidden -> R7 x0 head；
6. 冻结 Wan trunk 做短 probe，再决定是否解冻 full trunk。

推理时只输入 R7 anchor、text 和 noise，不加载 Wan VAE；bridge 和 R7 decoder 保留。

对照：

- 当前 random adapter；
- moment-matched/Procrustes linear bridge；
- teacher-distilled nonlinear bridge。

若 bridge 显著改善后半段运动，说明主问题是 representation language mismatch；若 bridge 也失败，再回到 R7 表示/codec。

### Anchor memory

T2V-1.3B 不自带 I2V image cross-attention。当前只把 clean anchor 拼进 self-attention 一次，条件注入不足。新 adapter 应：

- 在每个 Wan block 增加 gated anchor memory/FiLM；
- anchor 仍参与 self-attention，但未来 token 每层都可读取 anchor memory；
- 训练中加入 10--20% 轻微 anchor corruption/dropout 防止复制；推理使用 clean first frame。

## Phase 3：运动感知训练和 rollout

### Objective

主目标仍可使用 x0 + flow matching，但不再使用均匀全序列 MSE 作为唯一目标：

- horizon weight 初始为 `1, 1.5, 2, 3`，通过 probe 调整；
- motion direction cosine；
- motion magnitude ratio；
- acceleration；
- expanded geo motion；
- chunk 3/4 late-half 权重至少高于 chunk 1。

当前 auxiliary loss 只对 `t >= 0.6` 的样本生效，且主要是元素 MSE；新实现必须记录有效 mask 比例，并增加 cosine/magnitude 项。

### 短窗 overlap rollout

不要把“一次生成 4 个 future chunks”当作最终 1 秒视频方案：

1. 每窗只预测 2 个 future chunks；
2. 下一窗口使用上一窗口末端的真实/生成 context；
3. 通过 overlap 生成完整 1 秒；
4. 第二阶段再加入 scheduled sampling，从 0 增到最多 25%；
5. 先测 teacher-context 与 generated-context gap；gap 过大时修 bridge/decoder，不直接增加 rollout 比例。

## Hard gates

### Codec gate

- factor-1：PSNR >= 24.5、LPIPS <= 0.12、geometry motion cosine >= 0.95；
- t2 fallback：PSNR >= 23.9、LPIPS <= 0.13、geometry motion cosine >= 0.90；当前 0.832 不允许作为生产输入；
- 无 grid/checkerboard；boundary ratio <= 1.10。

### Generation gate

相对于 A baseline：

- future RGB PSNR 提升至少 3 dB，或 LPIPS 下降至少 20%；
- late-half LPIPS 不恶化；
- chunk 3/4 motion cosine >= 0.50；
- chunk 3/4 motion ratio 在 0.60--1.40；
- expanded geometry motion cosine >= 0.30；
- generated std ratio 在 0.8--1.2；
- 4 个 seeds 至少 3 个通过。

训练 checkpoint 的 `best` 必须同时参考 RGB composite 和这些 horizon guards，不能只看 x0 MSE。每次 eval 的图片和视频是人工审查的必要证据，不是可选装饰。

## Phase 4：全量扩展

只有 10k arm 通过 generation gate 才重建完整 SpatialVID-HQ cache：

- 新 representation/schema/cache version；
- 重新计算 exact CPU-FP64 per-position/channel statistics；
- 训练/评估 stats 与 representation signature 固化；
- 不复用 t2/c192 stats。

全量训练按 8k warm start -> 16k continuation -> 24k candidate continuation；每 1k 保存 PNG/MP4 和 64-clip eval。只有 hard-guard best 才可晋升。

## 明确禁止

- 不直接把当前 v1 盲目续到 30k；
- 不只增加 decoder robust steps；
- 不只提高 CFG；
- 不只看总体 latent/x0 MSE；
- 不用 `decoder_robust` checkpoint 替代 accepted `r7_ckpt`；
- 不复用不同 temporal factor/channel split 的 cache/stat；
- 不把 AE target 当作原始 RGB ground truth。

## 静态验证

```bash
python -m py_compile utils/video_preview.py \
  sample_causal_video_diffusion.py sample_causal_wan_video_diffusion.py \
  train_causal_wan_video_diffusion.py
bash -n scripts/scale/16_train_wan_t2v_diffusion.sh
git diff --check
```

以上检查不证明 NPU/HCCL/MoXing/OBS 正确。每个新 arm 仍必须先跑 Wan load、text sidecar、decoder decode、PNG/MP4 publication、resume smoke。
