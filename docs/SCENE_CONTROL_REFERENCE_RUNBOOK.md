# 固定窗口运动控制与观察外观参考实验

2026-10-23。入口为 `scripts/scale/56_train_scene_control_reference.sh`，配置为
`configs/scene_control_reference_v1.json`。三个节点、每节点八张 Ascend 910B。
本轮保持生产 AE、C256 表示、标准化、4 帧观察和 5 帧未来目标；三组分别从
stage53 `train/weights_step0031069.pt` 的 online 参数初始化，重新创建 AdamW
和 FP32 EMA。B/C 明确接收未来控制，不能把其结果称为无控制的自主未来预测。

## 要验证什么

| 组 | 观察条件 | 额外条件 | 问题 |
|---|---|---|---|
| baseline / A | F0–F3 latent | 无 | 相同样本与更新预算的继续训练对照 |
| motion / B | 同 A | 未来相机条件、稀疏点轨迹 | 明确运动后是否更容易生成清晰未来？ |
| reference / C | 同 B | 沿轨迹搬运的观察 texture，以及支持/置信信息 | 外观对应是否进一步减少轮廓和纹理崩塌？ |

这不是 stage54 的动态损失加权：三组使用相同的原始 x0 损失。新增控制作为
模型输入，不改 diffusion target，也不把搬运后的特征强行当作未来 latent。
不重训 AE，不增加 decoder RGB loss，不硬屏蔽动态 token 的跨帧 attention。

这一版先验证已有 StreamVGGT 能提供的相机与点轨迹。点查询 ID 是对应关系，
不是人/车类别，也不是语义实例 ID；尚未接入分割模型、dense instance mask、
骨骼动作、WASD 界面或运动模板库。需要先判断强运动条件是否能改善现有画面。

## 数据与条件边界

训练来自原 stage53 街景 32,768 窗口。默认确定性选择 4,096 个窗口，三组使用
同一份准备成功的样本集合，沿用原训练集统计量。`dataset.train_windows=0`
请求全部原窗口；改变数量需使用新 namespace，不能混入已有训练恢复。
验证沿用原 64 例开发集，不是新增独立测试集；中途固定 16 例。
如果坏验证视频被跳过，新输出 case 编号会压缩；每例保存 `source_eval_index`，
与历史结果对照应按 `id`＋`frame_ids`，不要只比较同名 case 编号。

准备过程结合 cohort 的原始 RGB 地址和源缓存的准确 `frame_ids`，与原 resize/
crop 约定对齐。从原始帧做教师推理，禁止把 RAW/AE/GEN 拼接预览 MP4 当作
训练标注。只按需下载使用中的视频到节点有界缓存，不拉取整个数据集。

教师复用现有完整 StreamVGGT checkpoint，代码使用 stage55 修复后的私有
vendor 导入。无需下载原版 VGGT，也不会访问 Hugging Face。Full-window
教师预测用来形成 **明确提供的未来控制**；prefix 教师用于观察期对应和尺度
约定。未来 RGB、未来 texture、未来 dense depth 不作为输入图；未来 depth
仅用于教师的稀疏动静态/有效性判断，这些标签仍属于未来辅助控制。
观察外观仅来自 F0–F3，传输是软条件；无法覆盖的新显露区域仍由 diffusion
生成。教师标签可能不准，稀疏首帧查询可能漏掉后入场人物和细肢体。

三组的数据曝光在准备阶段统一，坏原视频/无效窗口记录后跳过；不能给 B/C
临时替换样本而 A 继续使用原集合。身份不一致、空训练集、坏 checkpoint 等
程序有效性错误仍会终止，避免在错误数据上长训；画质、PSNR、motion mask
覆盖率等指标不作为阻断后续组的自动门槛。

## 启动

三节点使用同一入口，沿用原 ModelArts 环境和既有 StreamVGGT 路径。

2026-10-26 启动修复：若日志在计划协调器 `exit code=0` 后出现
`torchrun: error: ambiguous option: --local`，最终 `exit_code=2`，这是准备
worker 启动失败，尚未进入教师推理和三组训练。集群 Python 3.11 会提前解析
子脚本的 `--local`；入口已在 torchrun 和 Python 脚本之间补上 `--`。
三节点同步修复后的代码，保持原配置、namespace 和原启动命令重启即可；
既有计划/索引可以复用，无需删除输出。此时不能跳过准备直接用 `stage=train`。
本地已用官方 Python 3.11 argparse 和真实 torch 2.7.1 parser 对实际 all/smoke
命令复现并验证修复，9 项启动/协调器回归通过；尚未代替真实 NPU 运行验证。

先做真实模型与真实原始数据的小预演：

```bash
SCENE_CONTROL_STAGE=smoke bash scripts/scale/56_train_scene_control_reference.sh
```

预演使用独立 `scene_street_control_reference_v1_smoke*` 输出，从前 8 个源 shard
建立索引后选取最多 24 个训练窗口，另有 4 个验证例、4 步采样。这样无需先
扫描全部约 49 GB 源缓存。这个受限样本池不用于画质/数据代表性对照。
仍加载真实完整 AE/DiT/教师，仍使用全部 24 张卡。
每组先执行两个 optimizer update 并完整保存，再由新进程恢复并更新到第 3 步，
检查真实恢复链路。它能暴露 NPU 算子、显存、原始数据对齐、双写和恢复错误；
三步生成画质没有研究判别价值。

完整对照：

```bash
bash scripts/scale/56_train_scene_control_reference.sh
```

默认顺序：CPU/IO 建立控制准备计划 → 24 个独立教师 worker 并行准备 → 合并
同一份控制数据 → A 两步预演/新进程恢复训练 → B → C。每组使用全部 24 张卡，
后组不继承前组更新后的权重。每次启动相同命令可恢复；已完成部分按收据恢复。

可分段运行：

```bash
SCENE_CONTROL_STAGE=dry bash scripts/scale/56_train_scene_control_reference.sh
SCENE_CONTROL_STAGE=prepare bash scripts/scale/56_train_scene_control_reference.sh
SCENE_CONTROL_STAGE=preflight bash scripts/scale/56_train_scene_control_reference.sh
SCENE_CONTROL_STAGE=train bash scripts/scale/56_train_scene_control_reference.sh
```

`dry` 只显示计划，不执行 checkpoint staging、OBS 或训练。
`prepare` 只生成控制缓存。`preflight` 和 `train` 要求已有完整控制缓存；前者每组
两步，后者直接恢复/开始训练。烟雾预演始终用 `STAGE=smoke` 恢复，不能用正式
配置写入 `_smoke` 目录。只跑某组可设置 `SCENE_CONTROL_ARMS=reference`，但缺少
相同预算的 A/B 对照时不能归因于新条件。

配置和 namespace：

```bash
SCENE_CONTROL_CONFIG=/path/to/custom.json \
SCENE_CONTROL_NAMESPACE=scene_street_control_reference_v2 \
bash scripts/scale/56_train_scene_control_reference.sh
```

有效配置写到每节点 `launcher/nodeN/effective_config.json`；正式与预演分别固化。
修改同 namespace 的配置会提示使用新 namespace，以保护可恢复状态和对照身份。
`SCENE_CONTROL_SOURCE_NAMESPACE` 默认 `scene_street_prefix4_48h_v1`；源数据只读。

## 时间、显存与收敛

保留原 SceneFlow 主体，新增小控制投影；实际参数量、峰值显存和速度以预演
日志为准。教师只在离线准备阶段加载，每卡一份，训练阶段不与 DiT 同驻显存。
教师采样/深度/轨迹开销不能按旧纯缓存训练速度估计。

正式准备计划会扫描全部源 latent shard 的索引（`dataset.source_shard_limit=0`）。
首次可能读取约 49 GB 原缓存；它不将
整批缓存常驻内存/磁盘，并保留已扫描索引以便恢复。正式 4,096 窗口的 raw
教师处理在 24 卡并行，不能承诺固定准备时间。配置 `teacher.query_chunk_size`
可用于调整跟踪查询分块；改变它也应使用新 namespace 保持标签身份。

默认每卡 batch1×累积4，global batch96；主干 LR warmup200 至 5e-6，12,000 步降到
1e-6。B/C 新控制投影使用 `control_lr_multiplier=10`，对应峰值 5e-5，与主干
使用相同 warmup/衰减比例及统一梯度裁剪；A 不含新控制参数。每 1,000 更新
做配对采样与监督去噪检查，至少 4,000 步、连续 4 个检查
无超过 0.2% 的显著改善后结束，最早通常是 5,000 步。40,000 步是安全上限，
不是画质收敛证明。`hours=0/train_hours=0` 不设墙钟训练截止。

三组要比较相同更新数的快照；独立平台终点的训练曝光可能不同。不能仅凭
监督 MSE 平台、轨迹服从性或 RGB L1 宣称画质解决。固定窗口中人物身份、肢体
分离、遮挡、纹理、静止背景都要逐例检查。

## 输出、双读双写与空卡保护

默认持久镜像根为原个人 OBS `output/scale/`，同时写 ModelArts 当前输出根：

```text
scene_street_control_reference_v1/
  launcher/nodeN/pipeline.log
  launcher/nodeN/effective_config.json
  launcher/nodeN/npu_idle_guard/
  launcher/nodeN/exit.json
scene_street_control_reference_v1_controls/
  contract.json
  prepare/...
  data/cache_complete.json
  data/eval.pt
scene_street_control_reference_v1_baseline/
scene_street_control_reference_v1_motion/
scene_street_control_reference_v1_reference/
  contract.json
  train/checkpoint_latest.pt
  train/weights_stepNNNNNNN.pt
  train/full_stepNNNNNNN.pt
  train/convergence.json
  train/stopped.json
  eval/source31069_online/...
  eval/stepNNNNNNN/...
  complete.json
```

阶段 checkpoint、FP32 EMA、Adam、rank RNG/游标/坏 shard、平台状态按现有恢复
链路保存；每小时 latest/独立权重、每四小时完整快照、每次评估保留权重，结束
保存完整状态。中间控制产物和最终缓存带身份与内容收据，支持本地、当前输出、
持久镜像双读；双写降级会记录，两个持久根均失败不会冒充成功。

整个作业只启动一套空卡守护，覆盖 source staging、纯 CPU 索引、OBS 等待、
教师准备、组间转换和评估。默认利用率低于 2% 持续 600 秒后周期工作，全部
八张本地卡都在守护范围。守护不计入 `DI_throughput`。TCPStore 协调准备阶段，
不用 HCCL 等待长时间 CPU/IO；教师 worker 失败会通知并停止其他节点的准备
进程。后台守护/日志发布退出有界清理，避免完成后再空等两小时。

默认使用 `MASTER_PORT=29956` 做 torchrun；三个准备协调端口使用 29957–29959。
`SCENE_CONTROL_PREP_TIMEOUT_SECONDS` 默认 172800 秒，限制失联/异常漫长的准备，
不是画质门槛，可按实际网络预算调整。原训练的 HCCL 配置保持不变。

训练 DI 仍计每卡每秒未来 latent token，分母包含数据读取、保存和评估等待；
准备 DI/处理窗口数单独记录，不能把教师吞吐和训练 token 吞吐混在一起。

## 结果解释与后续

如果 B 明显优于 A，说明明确未来运动降低了当前任务难度；C 再优于 B，支持
继续研究观察外观的对应与传输。若强控制下仍然模糊，应回到 texture 生成与
decoder 误差，不继续只改轨迹格式。精确轨迹成功后再比较稀疏/编辑控制，并
为同一观察视频生成左/右/停等反事实运动，检验模型是否实际响应条件。

本轮尚未实现完整交互物理 world model；相机轨迹不是 agent 因果动作，指定
未来轨迹也不等于模型自己预测物体动力学。最终论文需要验证几何对齐相对
普通二维对齐/其他视觉特征的收益，而非仅展示 depth/track 条件可以输入模型。

本地 CPU、Bash 与协调器测试不等于真实 NPU/HCCL、完整 checkpoint、OBS 和
画质验证；完整链路的第一项集群动作应为上述独立 smoke。

本次本地回归共 85 项通过，包含实际两进程 Gloo 训练、连续训练与镜像恢复的
参数/优化器/EMA/RNG 一致性、原始目标保持、未来外观不变性、Bash 参数对接，
以及准备协调器失败/超时传播。RGB 重放单测使用模拟解码后端和实际生产裁剪；
本机缺少真实原视频/完整教师权重，不能以这些单测宣称真实教师标签可靠。
