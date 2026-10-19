# 固定窗口动态细节对照：按收敛训练

2026-10-19 实现。目标是在已有可辨识画面的 4 帧条件、5 帧未来基线上，定位并减轻人物轮廓和运动细节的快速崩塌。仍用三个节点，每节点 8 张 910B。用户已把“两天”改为按收敛决定时长；本轮默认不按墙钟时间截断。长视频、相机控制及增加数据覆盖留作后续独立实验。

当前状态：代码和本地 CPU/Gloo 测试完成；未提交真实集群训练，不代表 NPU 全形状或画质已经验证。入口为 [54_train_scene_motion_converge.sh](../scripts/scale/54_train_scene_motion_converge.sh)，配置为 [scene_motion_converge_v1.json](../configs/scene_motion_converge_v1.json)。

## 已有证据与尚未确定的原因

Stage53 已完成 31,069 步，最终 FP32 EMA 是当前街景短时续写参考。此次检查了 case004、011、007、024、048 的人物区域及未来逐帧对照：部分 AE 输出仍能区分腿部和身体边缘，GEN 在第一张未来帧已经出现双影、边界融合，后续加重。因此这些案例存在生成阶段新增误差，不能全部归因于 AE 分辨率。case041 的 AE 亮度本身仍有问题，不能把所有失败都归因于 diffusion。

这些人工选择区域仅用于诊断，不是训练标签或独立评测集。有的坏结果帧间变化反而更大；例如 case011 局部平均 RGB 帧差 RAW 约 .067、AE .056、GEN .089。因此“不动”的惩罚、最大化运动量或简单平滑都不是合适的主目标。证据在 `D:/workspace/VggAE_DataAudit/scene_prefix_20261019/motion_audit/`；原结果概况见 [stage53 结果](SCENE_PREFIX_RESULTS_20261019.md)。

目前还不能从这些图确定是局部训练权重不足、多模态未来的不确定性、geo/texture 预测不同步，还是 decoder 对特定误差敏感。本轮测试第一个可低成本隔离的因素，并加入分支替换诊断。若未改善，不再把同一损失换几个系数当作新的路线。

## 为什么测试这个因素

[MotiF](https://arxiv.org/html/2412.16153v1) 使用由光流构造的运动权重强化视频 diffusion 的动态区域；[Latent Temporal Discrepancy](https://arxiv.org/html/2601.20504v1) 则在 latent 中计算时序差异。本轮借鉴后者的低成本形式，但我们的 VGGT 派生特征不是其 VAE 表示，作用必须实测。坐标位置上的 latent 变化还可能包含相机运动、遮挡、纹理变化和编码上下文，不能称为独立物体运动或语义前景。

[FlowLoss](https://arxiv.org/html/2504.14535v1) 探索了光流目标，但增加开销且并未建立所有设置下的最终收敛质量提升；本轮不在每步加入 RAFT。[LPL](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Paper-Conference.pdf) 的 decoder 感知损失提供另一种可借鉴的方向，但主要是图像证据，需要单独衡量低噪声处的收益和成本。

这轮是工程诊断和受控实验，不声称动作加权是新方法，也不以此单独建立论文创新性。

## 两组只改变一项

启动脚本顺序运行 `control`、`motion`，每组都使用全部 24 张卡。两组都从同一个 **stage53 step31069 的 online `model`** 重新初始化 AdamW 和 FP32 EMA，不是让 motion 接着 control 训练。源 online 与 EMA 已较接近；旧 EMA 仍保留作结果参考。冻结 AE、统计量、样本次序、噪声 RNG、学习率和评估 seed 相同。

| 项目 | 设置 |
|---|---|
| AE | `scene_rae_c256_15day_v1/production/ae/best_joint.pt`，与 stage53 receipt/hash 一致 |
| 表示 | 18×18×256，每帧；geo128 + RGB texture128，非纯 VGGT token |
| DiT | 原 SceneFlow 831,305,344 参数，768×24 主干 + 1536×6 去噪头 |
| 初始权重 | `scene_street_prefix4_48h_v1/train/weights_step0031069.pt`，online |
| 数据 | stage53 32,768 个街景窗口，8,192 ID×4；直接读原 compact cache |
| 条件与目标 | 观察 F0–F3，只生成 F4–F8；无文字、位姿或未来 GT 条件 |
| 验证 | 原 64 个 heldout 开发例，不是新测试集；中途固定前 16 例 |
| 优化 | AdamW，betas=.9/.95，weight decay=0，梯度裁剪1，FP32 EMA |
| batch | 每卡1×累积4×24=96窗口 |
| 学习率 | warmup200至5e-6，cosine至第12000步的1e-6，此后保持 |
| x0/采样 | 原时间分布与 shift4.5；Euler32、CFG1 |
| control | strength=0，loss及预测梯度与原实现相同 |
| motion | strength=1，原始权重上限3，逐样本均值归一化 |

不重新训练 encoder/AE、不下载原始训练 RGB、不重新编码训练 latent，不用额外人工标注。沿用这批窗口是为了隔离变量，不代表 32,768 个窗口足以解决所有泛化问题；后续扩展真实时间窗口仍有价值。缓存坏输入跳过并记录，跨组比较时必须检查坏 shard 集合，不能仅因 seed 相同就声称实际曝光完全相同。

设归一化 clean latent 为 z，F4 的前一帧使用最后观察到的 z3，其余使用真实前一未来帧，仅构造监督权重：

```text
a[t,n] = log1p(sqrt(mean_channel((z[t,n] - z[t-1,n])²)))
r[t,n] = min(1 + strength * a[t,n] / max(mean_tn(a), 1e-4), 3)
w[t,n] = r[t,n] / mean_tn(r)
z_u = (1-u)*z + u*noise
loss = mean(w * ((x0_prediction-z) / max(u,0.05))²)
```

权重 detach，不增加条件输入，不耗用随机数。实际权重均值为1，范围在[1/3,3]内，静态区域仍有监督；不是追求更大的预测运动。目标相关权重会改变回归目标，尤其可能影响高噪声下的多模态平均，不能声称严格保持原 flow 最优解。不能拿 motion 的 weighted loss 与 control 的 loss 数字直接判胜。

## 如何按收敛停止

每1000个 optimizer update，对固定16例、固定高斯噪声做四个噪声等级 u=.25/.5/.75/.95 的 x0 验证，记录 EMA 的：

1. 全体未来 token 的无权 x0 MSE。
2. 高变化 token 的无权 x0 MSE。掩码是上述 log1p RMS 不低于每例均值且 >1e-4；空区域回退全局误差，并记录标志。

任意一项相对上一个显著最优值改善超过0.2%，耐心计数归零；小改善可以累积。两项连续4个有效检查点都没有显著改善，且至少训练4000步，才结束优化。缺失、非有限、重复或旧步数的结果不累计平台。第一次在1000步建立参考，最早通常到5000步停止。

这些是 **带噪真实未来的监督去噪探针**，并非纯噪声自由生成；真实未来不进入观察条件。它们只决定优化平台，不能保证人物、纹理或运动质量。每次同时保存自由采样的 online/EMA 视频；最终仍需完整观看固定案例。没有 PSNR/L1 或画质门槛阻止下一组运行。

`hours=0` 和 `train_hours=0` 关闭时间截止，仍保留40000步安全上限。触及上限明确写 `step_cap`，不能报告“已经收敛”。如果到上限仍明显改善，应重新评估预算后显式调整恢复方案，不要覆盖原配置混淆实验身份。按 stage53 平均速度估计，5000更新约6小时/组，但新增评估和存储开销会改变耗时；不能保证两组在48小时内结束。

必须同时比较两种视角：两组同更新数的保存快照用于公平比较；各自平台终点用于判断实际最佳可用质量。独立终点的训练曝光可能不同，不可称为等计算量对照。

## 评估与分支诊断

源模型先生成64例×2 seeds；中途每1000步评估16例×1 seed×online/EMA；最终64例×2 seeds×online/EMA。普通 MP4 的列名和帧号烧录在顶部：

```text
RAW | AE | GENERATED online/ema | COPY
```

前4帧是观察条件，后5帧才是生成；COPY 重复最后输入帧。视频头部标签不进入任何图像指标。

最终 seed101 的 case004、007、011、024、036、041、048 额外保存 `_GT_ASSISTED_ORACLE.mp4`：

```text
RAW | AE | GENERATED | GT GEO + PRED TEX | PRED GEO + GT TEX
```

其中 GT 指缓存真实 latent 的对应通道，并非深度真值。两个 hybrid 使用未来真值，仅作反事实诊断，不能作为生成视频或几何能力证据；混合不同来源分支本身也可能离开训练分布。记录分支 latent MSE、逐未来帧 RGB L1、逐转移 RGB 差异。单项改善只能帮助定位敏感性，不能证明唯一因果；RGB帧差也不是光流或几何一致性。

最终收据缺失、普通或 oracle 视频写出失败会标记 partial。再次启动补评估，已停止的 optimizer 不再多训练。源基线若缺例，会临时加载同一 source online 补齐再恢复当前模型，避免把训练后权重误标为起点。质量差不阻止后续组；产物缺失也不会冒充评估完整。

## 启动、双写与恢复

沿用原 ModelArts 三节点环境，每节点相同入口：

```bash
git pull origin ascend-910b
bash scripts/scale/54_train_scene_motion_converge.sh
```

默认每组先做真实形状两步预演，保存完整状态，再用新 torchrun 进程恢复接着训练。重复相同命令可恢复；已完成的组跳过。若明确只跑一个组：

```bash
SCENE_MOTION_ARMS=motion bash scripts/scale/54_train_scene_motion_converge.sh
```

单跑 motion 能得到候选结果，但没有 control 不能归因于加权。`SCENE_PREFIX_STAGE=preflight/train/all` 仍支持手动分段；默认 all。自定义配置用 `SCENE_MOTION_CONFIG`，改配置需新 `SCENE_MOTION_NAMESPACE`。两组各自固定源权重、AE、缓存、模型、标准化和配置身份，源输出只读。

持久输出在 ModelArts 当前输出根及原个人 OBS 镜像：

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/
  scene_street_motion_converge_v1_control/
  scene_street_motion_converge_v1_motion/
    contract.json
    eval/source31069_online/online/
    eval/stepNNNNNNN/{online,ema}/
    eval/stepNNNNNNN/summary.json
    train/checkpoint_latest.pt
    train/weights_stepNNNNNNN.pt
    train/full_stepNNNNNNN.pt
    train/convergence.json
    train/stopped.json
    train/bad-rNNN.jsonl
    launcher/nodeN/pipeline.log
    launcher/nodeN/exit.json
    complete.json
```

每小时 latest/独立权重、每4小时完整快照、每次评估独立权重，预演/结束完整保存。恢复包含 AdamW、FP32 EMA、逐 rank RNG/样本游标/坏 shard、收敛状态；停训决定与最终评估状态分开保存。本地快照仅在双根接受发布后清理，远端保留中间结果。评估文件有 hash receipts，缺失副本双读恢复；同步失败不能静默当成功。

原作业级空卡守护覆盖 CPU/OBS 和评估等待，低于2%持续600秒后周期工作，退出有界清理。训练 `DI_throughput` 表示每卡每秒处理的未来 latent token，分母包括 IO/保存/评估等待；另记录包含历史槽的网络 token 吞吐。评估吞吐只计自由采样 token，分母包含探针、oracle、解码和发布耗时。守护占卡不计入 DI。模型大小不变，新增权重是 [B,T,N] 小张量，无在线 RAFT/额外可训练网络；真实 NPU 峰值显存与速度由预演确认。

## 相机控制与下一步判断

相机轨迹可以减少背景投影和遮挡演化的不确定性，但不能唯一确定人物动作。连续相对位姿配合内参/射线适合模型内部表示；WASD/旋转可以是交互界面，在明确速度、时间步和坐标约定后积分成连续路径，无需单独人工标注每个 WASD 动作。可靠 RGB 时间对齐、尺度约定及有效性掩码仍需要处理。该设计是下一阶段提案，本入口尚未加入位姿或 WASD。

[CameraCtrl II](https://openaccess.thecvf.com/content/ICCV2025/papers/He_CameraCtrl_II_Dynamic_Scene_Exploration_via_Camera-controlled_Video_Diffusion_Models_ICCV_2025_paper.pdf) 可借鉴相机条件设计；[MotionCtrl](https://arxiv.org/abs/2312.03641) 区分相机与物体控制。二者不证明相机条件会自动修复我们的人物崩塌。控制加入后需同源、有/无轨迹的配对比较，再检验相机响应与画质；使用 VGGT 编码器本身也不等于已证明几何可控。

若候选在多数固定案例保住动态轮廓且静态细节不退化，保留并扩大真实窗口覆盖，再做轨迹控制。若公共探针改善但自由采样没改善，优先查高噪声误差与解码敏感性，而非继续优化该探针。若分支 oracle 明确指向某支预测误差，可据此设计低噪声 decoder 感知约束或分支特定目标；若两支替换都无稳定改善，重新审视联合表示和条件不确定性，不再堆相同权重实验。

本地验证包括真实小 SceneFlow 两进程 Gloo 更新、FP32 EMA、双写恢复、平台状态恢复、最终 oracle 缺失修复、源资产身份、loss/梯度退化到原实现、目标梯度隔离与 RNG、实际 Bash 分组调用和失败传播。它们不替代 Ascend/HCCL/MoXing/全形状验证，也不证明生成质量。
