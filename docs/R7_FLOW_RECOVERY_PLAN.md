# 几何 RAE flow 恢复实验（2026-09-07）

本轮目标是首帧＋文本条件的视频生成，**不换回 Wan VAE，不引入 4D 场景任务**。固定现有 geo112|tex80 R7，先建立可验证的 diffusion 闭环，再延长未来帧数。本文是实现与待运行协议，不是质量通过报告。

## 实测依据与修正

本地实验节选 `C:/Users/y50046448/Desktop/logs`：

- `r7_vggt_quick_geo112_tex80_v2/det_k1_n1/metrics.jsonl:61`：500步确定性单pair预测，PSNR vs AE=35.11、RAW=21.756、AE RAW=21.836。latent-only训练可以到达该pair的AE上限；不是表示完全不可解码。
- `det_k1_n16/metrics.jsonl:111`：16训练/16 held-out，RAW PSNR=19.510，AE=24.908；这是确定性预测，不是diffusion负结果。该目录flow只有2步smoke。
- 旧Wan7K/14K各4个保存样本，future仅使用训练位置/通道均值就得到motion cosine约 `[.996,.921,.602,.241]`，几乎复现模型。旧t2首段高分受anchor/tail均值偏移主导，不能解释为学会运动。
- 同4样本Wan7K/14K raw std ratio约1.026/1.000，但标准化空间std ratio约.590/.500。该比值是张量内部变化尺度，不是多seed条件多样性。
- 旧Wan14K EMA eval t=.9 x0 MSE=.2913，未经训练的 `x_t/t` 理论MSE约.01235。所有t bucket都随训练改善，但近干净端仍不足。
- 历史EMA默认.9999且无warmup；如未覆盖默认，7K/14K仍保留49.7%/24.7%初始参数贡献。缺少online对照，不能据此直接判过拟合或断言EMA是唯一原因。
- 旧t2样本实际RGB decoder temporal_blocks=0，最新geo112 probe为2；不得混用故障解释。
- manifold同方向/尺度球面扰动相对Euclidean平均低.072dB；真实端点geodesic插值的小收益不证明球面diffusion适合生成。

## 新实现

- `utils/latent_generation_metrics.py`：train-only FP64统计、可逆zscore、均值基线、raw/centered差分、实际解析去噪基线。
- `models/r7_flow_probe.py`：共享小型时空Transformer，显式首帧条件、可选UMT5 text cross-attention、逐block时间调制；两种独立head契约。
- `train_r7_flow_probe.py`：冻结R7，materialize一次，释放大encoder；train-memory/held-out分别评估，online/EMA分别评估，多seed采样，完整恢复。
- `scripts/scale/25_run_r7_flow_probe.sh`：独立diagnostic namespace，node0/device0运行，其余节点退出不等待collective；不提交48份重复小样本训练。
- `scripts/scale/smoke_r7_flow_probe.sh`：两种head各2步，每种在第1步保存并退出、再恢复到第2步；只证明运行/恢复链路。

### 旧Wan online/EMA重放

新增 `evaluate_r7_wan_denoising.py` 和 `scripts/scale/24_replay_r7_wan_denoising.sh`。
必须显式指定原实验checkpoint、匹配manifest、text sidecar；不会猜测或替换成新实验。
同时评估online/EMA、各t实际解析基线、CFG1/3、30/60步，可选uniform积分网格、oracle加噪起点和条件打乱。重放保持checkpoint中的旧时间/anchor-memory语义；不把旧权重按新timefix解释。缺失权重或未知契约会失败。

```bash
REPLAY_CHECKPOINT=obs://.../original_run/checkpoint_step0007000.pt \
REPLAY_MANIFEST=obs://.../matching_cache/eval/manifest.txt \
TEXT_EMBEDDING_DIR=obs://.../text_embeddings/umt5xxl_spatialvid_10k_v1 \
REPLAY_R7_CKPT=obs://.../original_codec/joint/checkpoint_best.pt \
REPLAY_DECODER_CKPT=obs://.../original_codec/decoder_robust/checkpoint_best.pt \
REPLAY_NAMESPACE=r7_wan7k_replay_diag_v1 \
bash scripts/scale/24_replay_r7_wan_denoising.sh
```

不提供R7 checkpoint时只计算latent诊断；提供时严格核对decoder契约。缓存没有future原RGB，输出只能叫AE_TARGET。
`ORACLE_STARTS=0.1,0.5,0.9`仅用于定位，不算自由生成；`CONDITION_ABLATIONS=1`需要至少2个eval clips。

### 两个head，不混用loss

约定：`x_t=(1-t)*epsilon+t*z`，z在冻结train统计的标准化空间。

1. `plain_x0`：直接预测z，普通x0 MSE。
2. `preconditioned`：令 `d=t²+(1-t)²`，
   - 网络输入 `x_t/sqrt(d)`；
   - `z_hat=(t/d)*x_t+((1-t)/sqrt(d))*F`；
   - 稳定网络目标 `((1-t)*z-t*epsilon)/sqrt(d)`；
   - 稳定速度 `((2t-1)/d)*x_t+F/sqrt(d)`。

后者是加权x0目标，不与plain loss数值直接比较。它在t=1具有输入保留性质，但不能只凭端点指标宣布成功；必须测强噪声、自由采样与解析基线。初期uniform t、uniform Euler，不同时加RGB/运动辅助loss。

### 前缀解码

K=1/2/4/8对应连续未来帧[1..K]。完整输入为真实anchor+K个候选+重复最后候选补齐9帧，只评分前K帧。不用真实未生成suffix。AE-prefix与copy使用相同契约，完整真实9帧AE另列oracle。K=8即完整窗口。

### 条件、统计与恢复

- n1默认无文本，用于记忆管线；n16+默认要求UMT5 sidecar。仅显式`ALLOW_NO_TEXT=1`才做无文本诊断，不得声称首帧＋文本生成。
- n1统计来自该pair空间token，只用于记忆，不是总体数据分布估计；每个样本数/窗口/head独立namespace。
- checkpoint冻结实际materialized latent/RGB/text hash、video/window IDs、codec签名、stats和objective；resume改变任一项拒绝。
- FP16按实际resolved dtype启用GradScaler，overflow明确记录并降低scale，不推进scheduler/EMA/optimizer步数；连续16次失败终止。
- EMA warmup，online为小集合主评估，同时报告EMA；不能仅选EMA曲线解释泛化。
- 文件输出latest/best/periodic/final、TensorBoard、JSONL、PNG/MP4。旧输出无明确resume禁止覆盖。

## 执行顺序

先在模型机器运行tensor测试与smoke；Windows无torch只可做静态/公式检查。

```bash
bash scripts/scale/smoke_r7_flow_probe.sh

# 默认只比较n1两种head。不会自动启动n16或长视频。
STAGE=n1 FLOW_NAMESPACE=r7_geo112_flow_probe_v1 \
  bash scripts/scale/25_run_r7_flow_probe.sh
```

n1默认2000步、4 seeds，记忆验收为所有训练样本/seed的PSNR vs同契约AE>=30dB、RAW PSNR距AE<=.5dB。此阈值仅是管线记忆验收，不能要求held-out开放式生成匹配唯一真实未来。

查看`memory_status.json`、`eval_samples.jsonl`、`denoising.jsonl`和预览后，手动选择head（以下仅示例，不预设赢家）：

```bash
STAGE=n16 PREDICTION=preconditioned \
  N1_STATUS=obs://.../n1_preconditioned_n1_f1/memory_status.json \
  TEXT_EMBEDDING_DIR=obs://.../text_embeddings/umt5xxl_spatialvid_10k_v1 \
  FLOW_NAMESPACE=r7_geo112_flow_probe_v1 \
  bash scripts/scale/25_run_r7_flow_probe.sh
```

短窗必须再提供前一级同codec/head/sample-count记忆报告：

```bash
STAGE=prefix PREDICTION=preconditioned FUTURE_FRAMES=2 MAX_SAMPLES=16 \
  N1_STATUS=obs://.../n1_preconditioned_n1_f1/memory_status.json \
  PRIOR_STATUS=obs://.../n16_preconditioned_n16_f1/memory_status.json \
  TEXT_EMBEDDING_DIR=obs://.../text_embeddings/umt5xxl_spatialvid_10k_v1 \
  bash scripts/scale/25_run_r7_flow_probe.sh
```

K=4要求K=2报告，K=8要求K=4报告。**gate仅证明训练记忆，启动下一阶段仍须人工检查held-out与运动**。不自动重建10K/全量cache，不写生产gate_passed。

同namespace恢复要求显式`PREDICTION`与`RESUME`，所有训练参数必须不变。只支持同预算恢复，不隐式延长训练；实验扩展需新设计、新namespace。

## 报告解释

- `raw/*`、`mean_only/*`与`position_centered/*`并列，去均值cos也不是物理运动指标，尤其首段共享真实anchor。
- `seed_pair_mse`是同条件不同seed；`same_noise_condition_pair_mse`是同seed不同条件，不能用前者替代后者。
- n1收敛到同一目标是成功，不要求n1多样性。
- `full_ae_psnr_raw`是oracle ceiling，不是生成指标；没有future原视频的旧cache重放只能标AE_TARGET。
- `best`按online train-memory normalized MSE保存；held-out与RGB要一起审查，生产质量不自动晋升。

## 验证与边界

本地：`python scripts/test_r7_flow_probe.py`，`python -m py_compile ...`，`bash -n ...`，`git diff --check`。若无torch，tensor测试必须打印SKIP，不得称模型验证通过。NPU/HCCL/OBS/LPIPS及恢复smoke需在真实集群执行。

本轮不做：重新训练codec、Wan VAE、4D场景、warp/scene flow、Riemannian diffusion、盲目延长旧长跑、放宽生产gate。
