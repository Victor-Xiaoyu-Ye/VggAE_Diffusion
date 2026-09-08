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

## 2026-09-07 晚间 n1 更新与下一轮

`r7_geo112_flow_n1_probe_v1/n1_plain_x0_n1_f1` 提供的记录到500步online评估：
RAW PSNR12.823、AE PSNR21.836、generated-vs-AE12.893、normalized MSE1.333。
t=.1/.5/.9的去噪MSE为.2232/.0531/.0409。去噪有改善，自由采样记忆未通过。
这证明存在去噪/采样差距，但**不证明off-manifold、速度误差放大或某个参数化是唯一原因**。
单次train loss变化还混合了随机t，新增日志记录t，不能只归因随机noise。

用户报告平台显示成功，但节选没有final checkpoint和500步EMA评估；退出原因未知。
`.tmp-*`缺失是后台watch捕获的竞态，不会直接终止trainer；同步的临时文件不证明源进程必然被kill。
旧正式checkpoint仅验证ZIP目录可解析，未在本机加载torch/NPU权重。

本轮契约升级为 `r7-prefix-flow-v2`：保留旧实验，不把v1 checkpoint静默恢复到新目标/诊断配置。
三种head参数化、固定噪声但跨时间的path、oracle starts和完成标记都显式记录。

## 新实现

- `utils/latent_generation_metrics.py`：train-only FP64统计、可逆zscore、均值基线、raw/centered差分、实际解析去噪基线。
- `models/r7_flow_probe.py`：共享小型时空Transformer，显式首帧条件、可选UMT5 text cross-attention、逐block时间调制；三种独立head契约。
- `train_r7_flow_probe.py`：冻结R7，materialize一次，释放大encoder；train-memory/held-out分别评估，online/EMA分别评估，多seed采样，完整恢复。
- `scripts/scale/25_run_r7_flow_probe.sh`：独立diagnostic namespace，node0/device0运行，其余节点退出不等待collective；不提交48份重复小样本训练。
- `scripts/scale/smoke_r7_flow_probe.sh`：三种head各2步，每种在第1步保存并退出、再恢复到第2步；只证明运行/恢复链路。

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

### 三个head，不混用loss

约定：`x_t=(1-t)*epsilon+t*z`，z在冻结train统计的标准化空间。

1. `plain_x0`：直接预测z，普通x0 MSE。
2. `preconditioned`：令 `d=t²+(1-t)²`，
   - 网络输入 `x_t/sqrt(d)`；
   - `z_hat=(t/d)*x_t+((1-t)/sqrt(d))*F`；
   - 稳定网络目标 `((1-t)*z-t*epsilon)/sqrt(d)`；
   - 稳定速度 `((2t-1)/d)*x_t+F/sqrt(d)`。

3. `direct_velocity`：目标 `z-epsilon`，网络直接输出速度；采样不除以`1-t`，仅在评估时用 `z_hat=x_t+(1-t)*v_hat` 重建x0。

预条件化是加权x0目标，不与plain loss数值直接比较。它在t=1具有输入保留性质，但不能只凭端点指标宣布成功；必须测强噪声、自由采样与解析基线。初期uniform t、uniform Euler，不同时加RGB/运动辅助loss。

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

在同一ModelArts代码目录/原有环境准备之后执行（必须同步本轮代码，不能只提交旧分支）：

```bash
set -euo pipefail
export PYTHON_BIN=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
export VGGAE_REF_ROOT=/cache/yexiaoyu/vggae_ref
export STREAMVGGT_CKPT="${VGGAE_REF_ROOT}/StreamVGGT/checkpoints.pth"
export R7_NAMESPACE=r7_t1_c192_geo112_tex80_probe_v1
export FLOW_NAMESPACE=r7_geo112_flow_n1_diagnostics_probe_v2
export MAX_STEPS=2000 EVAL_EVERY=100 DIAGNOSTIC_EVERY=500
export SAMPLE_STEPS=30 DIAGNOSTIC_SAMPLE_STEPS=60
export SAMPLE_SEEDS=42,43,44,45 ORACLE_STARTS=0.5,0.7,0.9
export HIDDEN_DIM=384 DEPTH=4 NUM_HEADS=6 BATCH_SIZE=1
export LEARNING_RATE=2e-4 WARMUP_STEPS=100
export LOG_EVERY=10 SAVE_EVERY=500 EVAL_LPIPS=1 EVAL_EMA=1
export RUN_SMOKE=1 RUN_FIXED_PATH=auto
bash scripts/scale/26_run_r7_n1_diagnostics.sh
```

前提：仓库已在当前目录，StreamVGGT由既有ModelArts准备命令放到上述位置，torch/torch_npu及项目依赖已安装，OUTPUT_URL由平台注入或使用配置的持久OBS输出根。脚本会stage已完成的R7 checkpoint和数据split，不会下载新模型或重建全量cache。

Stage26依次执行三种head的两步保存/恢复smoke，然后preconditioned与direct_velocity各2000步。两个random-noise n1均未通过时才执行plain-x0 fixed-path诊断；固定seed42的noise、循环32个t，seen-noise与unseen seeds43..46分开报告。不启动n16。`RUN_FIXED_PATH=0`关闭fallback，`RUN_FIXED_PATH=1`强制执行。

单独运行可使用stage25，例如：

```bash
STAGE=n1 PREDICTION=preconditioned \
  FLOW_NAMESPACE=r7_geo112_flow_n1_preconditioned_probe_v2 \
  MAX_STEPS=2000 STOP_AFTER_STEPS=0 SAMPLE_STEPS=30 DIAGNOSTIC_SAMPLE_STEPS=60 \
  ORACLE_STARTS=0.5,0.7,0.9 SAMPLE_SEEDS=42,43,44,45 \
  bash scripts/scale/25_run_r7_flow_probe.sh
```

`SAMPLE_STEPS`必须单整数，不支持`30,60`；逗号列表属于`DIAGNOSTIC_SAMPLE_STEPS`。普通采样与额外步数对照是同noise输入，额外步数/oracle只用第一seed，节省诊断成本；主gate仍要求全部四个seed。`sampling_diagnostics.jsonl`记录额外对照，不能替代正式gate。

默认n1门槛是全部训练样本/seed的generated-vs-AE PSNR>=30dB、RAW PSNR距AE<=.5dB，**没有新增std_ratio硬门槛**。这是记忆检查，不是视频泛化要求。`run_status.json`的completed仅表示指定训练步数完成；quality_gate另列。SIGKILL/节点掉电无法由Python写失败，但wrapper在可运行时捕获非零退出码；整个作业被杀可能保留running状态，不能视为完成。

查看`memory_status.json`、`eval_samples.jsonl`、`denoising.jsonl`和预览后，手动选择head（以下仅示例，不预设赢家）：

```bash
STAGE=n16 PREDICTION=preconditioned \
  N1_STATUS=obs://.../n1_preconditioned_n1_f1/memory_status.json \
  TEXT_EMBEDDING_DIR=obs://.../text_embeddings/umt5xxl_spatialvid_10k_v1 \
  FLOW_NAMESPACE=r7_geo112_flow_probe_v2 \
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
- `best`保留按online normalized sample MSE选取；另存`best_sample_rgb`（固定主采样seed中最差PSNR最佳）和`best_denoising`（固定t的online x0 MSE平均最低），选择依据不同，不能只看一个best。所有checkpoint同时含model/EMA，评估上述best选择时应使用online model。
- 目录同步通过`utils/output_snapshot.py`创建临时快照，排除`.tmp-*`/`.pending.*`及快照子树；已完成checkpoint用hardlink固定inode（不支持则复制），JSONL复制固定长度。remote目录输入仍走原下载路径。只清理本进程创建的快照，不删除源/远端临时文件。快照是逐文件稳定视图，不是整个训练状态的多文件事务。

## 验证与边界

本地：`python scripts/test_r7_flow_probe.py`，`python -m py_compile ...`，`bash -n ...`，`git diff --check`。若无torch，tensor测试必须打印SKIP，不得称模型验证通过。NPU/HCCL/OBS/LPIPS及恢复smoke需在真实集群执行。

本轮不做：重新训练codec、Wan VAE、4D场景、warp/scene flow、Riemannian diffusion、盲目延长旧长跑、放宽生产gate。
