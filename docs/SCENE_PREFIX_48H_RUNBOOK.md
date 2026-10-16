# 两天街景短历史续写实验

本轮把 3 节点 × 8 张 910B 的两天预算集中在一版可观看的短时续写：固定现有 SceneAE C256，从已检查过的 71,609 步 online 权重出发，只用街景缓存，观察前 4 帧，生成后 5 帧。先提高条件可辨识度和训练稳定性，再判断是否值得继续做单帧生成及几何控制。

这是画质优先的适配实验，同时改变了历史长度、抽样域和优化状态，不能用它单独证明某个因素的因果贡献，也不是完整论文消融。实现于 2026 年 10 月 16 日；本地验证通过，真实 Ascend 运行和生成效果以此次集群结果为准。

## 为什么选这版

历史回放中，video 71,609 online 的场景结构明显好于同一步旧 BF16 EMA；最终 137,104 online 已严重退化。详情见 [保存权重回放](SCENE_RAW_EMA_RESULTS_20261012.md)。71,609 是已验证的恢复候选，不声称是所有快照的全局最优。

单张图通常不足以确定相机行进方向和动态物体速度。连续四帧能直接提供短时运动证据，因而这次不要求网络同时猜运动意图和完成外观去噪。已有 AE、主干容量和空间分辨率保持不变，避免两天预算被重新编码与从零训练消耗。

街景收窄有助于减少任务差异，但“混合数据导致全部失败”仍没有被证明。本轮不增加相机控制：现有 SpatialVID latent 中 rays 为零，尚无可靠逐帧姿态与 RGB 对齐。

## 固定配置

| 项目 | 本轮设置 |
|---|---|
| 资源 | 3 节点 × 8 张 910B，同一分布式任务 |
| 表示 | 原生产冻结 SceneAE C256，18 × 18 tokens；原均值与标准差 |
| DiT | 原 SceneFlow，831,305,344 参数；768 宽 24 层主干，1536 宽 6 层去噪头 |
| 初始权重 | `scene_rae_c256_15day_v1/production/video/weights_step0071609.pt` 的 `model` |
| 优化状态 | 新 AdamW，betas 0.9/0.95、weight decay 0、梯度裁剪 1；新 FP32 EMA |
| 学习率 | 前 200 更新 warmup 至 1e-5，按更新数 cosine，到第 30,000 更新为 2e-6，之后保持 |
| 总更新上限 | 40,000；时间上限先到则保存收尾 |
| 有效 batch | 每卡 1 × 累积 4 × 24 = 96 窗口 |
| 训练数据 | 已审核 8,192 条 street ID × 原有 4 个窗口，预期 32,768 窗口 |
| 条件 | 原 F0 anchor，加 F1–F3 的干净 latent；无文本、无未来相机条件 |
| 目标 | F4–F8，共 5 个未来帧；x0 flow loss，仅对未来计算 |
| 采样 | Euler 32 步，固定 ID/帧索引/seed 的噪声，CFG 1 |

原窗口是约一秒的 9 帧，名义播放 8 fps。因此这版只生成约 0.625 秒未来，不是 5 秒视频，也不是原来的单图 I2V。原缓存没有保存 PTS，结果会保留原 `frame_ids` 和 cohort fps，并标明时间精度限制。

训练 ID 来自原选择文件的 `role=real_street_video_candidate`，不按模糊的运动标签猜测。原来的 256 条 appearance replay 和其余六域均排除。该集合已被历史模型看过，不能用于宣称从零数据效率。验证沿用原 heldout val，最多 64 个街景；不把训练 ID 重新划为“未见测试”。

## 数据流与预演

保留原 SceneFlow 的八个位置槽，避免时域位置重编号：

```text
原 latent 缓存 z0…z8，使用原统计归一化
  z0                         → 原 anchor memory
  clean z1,z2,z3 + noisy z4…z8 → 原 8 槽 SceneFlow
  输出的最后 5 槽              → x0 loss / flow 采样积分
  observed z0…z3 + sampled z4…z8 → 冻结 decoder → RGB
```

每个积分步重新拼入干净的历史三槽，ODE 只更新五个未来槽。未来 GT latent/RGB 只用于监督或采样完成后的 oracle/指标，不能进入生成条件。解码器即使含时域注意力，也只能看到真实历史和预测未来。

启动后自动完成以下步骤：

1. 从原只读输出中固定 AE、cache、flow checkpoint 的 receipt/hash。各节点只由 local rank 0 暂存大型公共权重，其余进程共享 mmap。
2. 用真实 StreamVGGT 检查未来帧扰动和截短输入是否改变历史特征；缓存量化差异只报告，不作为画质门槛。保留原来的 4 个街景验证窗口，再从原 val 补编码到最多 64 例；坏 RGB 记录并跳过。
3. 24 个 rank 互斥扫描旧的 2,184 个混合 shard，只重打包 street latent。无需重新编码训练 RGB；扫描、双写和各 rank 游标可恢复。坏 shard 单独记账；输出无法持久化不能伪装成坏输入跳过。
4. 保存同条件的 source 71,609 起点视频，完成真实形状的两步训练、全状态保存。
5. 启动新的 torchrun 进程，从这两步的 optimizer、FP32 EMA、RNG、数据游标和坏 shard 集合继续训练。预演更新计入本轮，不再重置模型。画面差或指标差不会阻止后续训练。

预演前也可能有较长的 CPU/OBS 准备时间。空卡守护覆盖整个作业，仅当利用率低于 2% 持续 600 秒后周期性工作；守护工作不进入 DI_throughput。退出沿用有界 TERM/KILL 清理，保存日志尾段和 `exit.json`。

## 时间和保存安排

总内部预算 46 小时，外部两天预算留约 2 小时给收尾余量。训练时钟最多 38 小时，包含该阶段的保存和中途评估；若准备较慢，总预算会先截断训练。总计时包含启动准备，续跑读取已消耗时间，不重新获得 46 小时。

训练在总预算剩两小时或训练时钟到期时收尾，保存最终 checkpoint，再做最终评估。评估也检查总预算，到期保留已完成案例并标记 `partial`。这不是对 OBS 带宽或 NPU 单步耗时的保证，实际步数由此次吞吐决定。

- 每小时保存 `checkpoint_latest.pt` 和独立 online/EMA 权重；完整状态含 optimizer、逐 rank RNG/游标及坏 shard 集合。
- 每四小时、预演结束和训练结束保存独立完整 checkpoint。
- 每两小时评估固定 8 个开发例，各输出 online 和 EMA；起点评估最多 64 例 × 2 seeds，结束评估最多 64 例 × 2 seeds × 2 套权重。
- 保存梯度裁剪前范数、分噪声区间误差、prediction std、输出权重范数、EMA 距离、显存和 DI_throughput。`DI_throughput` 为每卡每秒处理的未来 latent tokens，包含 IO/保存/评估等待；另记包括历史槽的网络 token 吞吐。
- 每个案例保存 MP4、生成 latent、指标和文件 hash receipt，可从两个输出根恢复并修复缺失副本。双写失败会显式记录，不伪报成功。
- 远端保留阶段快照；确认两个输出根接受本次发布后，本地仅保留最近两份完整快照和两份权重，另保留 latest。只删除本次任务自己的暂存和快照，不删除旧源输出。

## 启动与续跑

使用此前三个节点的 ModelArts 启动环境，保留原依赖准备和 StreamVGGT checkpoint。每节点运行相同入口，由现有 `VC_WORKER_*` 环境解析 rank：

```bash
git pull origin ascend-910b
bash scripts/scale/53_train_scene_prefix_48h.sh
```

默认新输出命名空间为 `scene_street_prefix4_48h_v1`，源生产目录保持只读。默认 `SCENE_PREFIX_STAGE=all` 自动预演后训练；相同命令可续跑，已完成的准备阶段跳过。若手工分开运行，可分别设置 `SCENE_PREFIX_STAGE=preflight` 与 `train`。不要用旧的 `SCENE_STAGE` 控制本入口。

配置在 [scene_prefix_48h_v1.json](../configs/scene_prefix_48h_v1.json)。改变数据、模型、历史长度或训练配置应使用新 `SCENE_PREFIX_NAMESPACE`，不能混入旧身份。这里只保留保证数据/恢复语义正确的检查；没有 PSNR、L1 或重建质量闸门。

持久输出：

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/scene_street_prefix4_48h_v1/
  contract.json
  prepare/prefix_contract.json
  data/cache_complete.json
  data/eval.pt
  eval/source71609/online/case000_seed101.mp4
  eval/stepNNNNNNN/online/case000_seed101.mp4
  eval/stepNNNNNNN/ema/case000_seed101.mp4
  eval/stepNNNNNNN/summary.json
  train/checkpoint_latest.pt
  train/weights_stepNNNNNNN.pt
  train/full_stepNNNNNNN.pt
  launcher/node0/pipeline.log
  launcher/node0/exit.json
  complete.json
```

同时写 ModelArts 当前输出根。视频四列依次为 RAW / AE / GENERATED / COPY。GENERATED 和 COPY 的前四帧直接显示输入 RGB，只有后五帧能证明生成效果；COPY 在未来一直重复最后一张输入帧。

## 怎么决定下一步

预期收益是同一场景短时未来更完整、纹理更少破碎、接续更连贯。两天不能承诺长视频、动作控制或通用世界模型。冻结 AE 保住已知的重建能力，但如果 flow 持续输出偏离可解码区域的 latent，画面仍会失败。

先在全部固定开发例上并排看起点、online、EMA，重点看未来的第 1/3/5 帧和输入接缝；同时看失败比例、运动是否与历史相容、是否退化为静止复制。RAW L1、copy L1、motion、Laplacian 只是诊断，低 L1 或高锐度都不能单独决定画质；只取少数好看样本也不能算稳定改进。

若多数样本明显改善且存在非静态的合理续写，再做共享目标和相同训练曝光的 1 帧对 4 帧控制实验，分离历史条件的作用。若只能保持静止外观，优先补可靠运动/轨迹对齐数据。若短时、有历史、窄域也普遍碎裂，就把预算转向表示到生成器的匹配和 decoder 对生成误差的容忍度，而不是再次直接扩大数据和训练天数。

本地检查覆盖真实小 SceneFlow 训练、两进程 Gloo、连续与恢复训练逐张量一致、镜像恢复、未来 GT 替换不影响生成、缓存重打包及实际 Bash 清理流程。它们验证实现语义和恢复链路，不替代这次完整形状的 Ascend 预演与观看结果。
