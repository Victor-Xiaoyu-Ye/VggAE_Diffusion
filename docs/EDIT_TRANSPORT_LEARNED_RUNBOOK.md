# Stage64：训练可复用的跨帧编辑传输模块

本轮将 Stage63 的逐视频求解改为可学习的编辑残差预测。目标是保留明确的首帧编辑，
利用源视频表示和对应关系修复跨帧传输，而不是把完整编辑再次投影回源视频附近。
这是完整效果实验，不是已经成立的几何优势或论文结论。

## 启动方式

保留现有三节点、每节点8张910B的启动方式。代码和现有 `vggae_ref` 同步完成后，
三台节点执行同一段命令：

```bash
LEARNED_STAGE=smoke bash scripts/scale/64_train_edit_transport.sh &&
LEARNED_STAGE=all bash scripts/scale/64_train_edit_transport.sh
```

正式输出命名空间 `edit_transport_learned_v1`；smoke 使用独立的
`edit_transport_learned_v1_smoke`，不会覆盖正式训练。
不增加 bootstrap，不在线安装依赖，不下载新的基础模型。

本轮只读复用：

- Stage60 `vace_edit_pilot_v1` 的原生编辑 bank，实际曾完成2048 train /24 dev；
  以当前 bank 中可恢复的记录为准。旧 target RAE token **不是输入特征**。
- 已完成 Stage63 `edit_reference_video_v1` 的24个合法生成/用户参考及源数据；
  不重新选择参考，不拿 dev target 首帧替代失败参考。
- 生产 Scene C256 RAE、StreamVGGT、`Wan2.1-VACE-14B` 既有权重。

## 训练的对象

| 模块 | 本轮状态 |
|---|---|
| StreamVGGT | 冻结，提取已观察源视频和合法参考的表示 |
| Scene C256 RAE | 冻结，其 decoder 参与反向传播到编辑传输模块 |
| 编辑传输网络 | **训练**，权重跨样本复用；保存模型、优化器、EMA和恢复状态 |
| VACE14B | 冻结，只在后续完整视频渲染阶段加载 |

默认编辑传输网络为 **2,774,656 个参数（约2.77M）**：宽192、4层，包含局部空间
混合、时间注意力及首帧参考交叉注意力。并非再次从头训练一个14B视频生成模型。

模块输入包括源 RAE 状态、首帧参考表示、RGB传输提议、源对应置信度及允许编辑的
支持区域。预测 latent 残差，经冻结 decoder 转换成可见 RGB 修正。
输出保留直接 RGB 提议旁路；零初始化时等于 RGB 提议，传输参考的首帧强制保留合法参考。
可靠可见区域保留提议，主要在允许编辑区域内的不确定对应上修正。
后续原生 VACE 渲染不是逐像素硬拼接，因此最终成片的首帧不保证与参考逐像素相等。

源光流置信度和渲染允许区域分开：不能因为对应不确定，又在最终写回把新预测的
补全全部乘零。源光流仍是经典 DIS，不声称它是语义分割或几何真值。
九时刻与17帧源光流的允许支持取并集并共享给三路，避免两套对应的差异再次屏蔽
补全。首帧硬保持只适用于中间参考；最终VACE仍是条件生成，没有偷偷拼接真值帧。
同样不能认为源光流自然解决了新物体、拓扑变化或大幅参考形变。

这与之前的 RAE 辅助 loss 不同：模块输出的编辑状态实际进入最终 VACE 写回，
可在中间参考和最终视频中分别检验。训练目标不代表必然获得更好的成片。

## 数据协议和收敛

训练只使用 Stage60 原 train 集，并按源 group 确定性留出约6.25%作为训练监控集。
原 dev 用于完整模型参考视频比较；原保留 test 和 benchmark_dev 不进入训练。
原数据划分不是经过外部验证的场景完全无重叠 benchmark，不能据此夸大泛化结论。

训练中 `target[0]` 模拟用户已经编辑的一帧参考；后续 edited target 只作为监督，
与模型输入分开保存。不能使用未来 target 生成条件 mask、源 token 或光流。
这属于参考编辑训练协议，不等同于纯文本编辑。测试时参考来自已记录的 Stage63
前端或明确用户提供的参考；前端改错对象仍应记录为端到端失败。

默认训练配置：

- 最多3000个训练step，最少500步后再依据监控结果判断停训；失败跳过不计入实际
  参数更新次数，日志分别记录 step 和 updates。
- 每250步监控，连续4次没有达到配置要求的改善后可结束；每125步阶段保存。
- 学习率0.0002，warmup50步；EMA0.99，梯度裁剪1.0。
- 10% identity 样本约束不编辑时的稳定性；30%概率对训练输入的传输观测添加
  时空连续缺口，置信度随缺口置零，保留源视频和首帧参考。该增强不读取监督，
  用于学习不确定区域补全，评估时关闭。padding 不计入 RGB 监督指标。
- 每卡每次一个样本，24卡同步；实际有效批量和采样策略以训练记录为准。
- 训练监控集只从 train 中产生，不用24例 dev 的最终表现挑参考或反复选择 checkpoint。

以上是带最大预算的收敛策略，不是3000步必然收敛或画质足够的承诺。
数据失败应记录并跳过；局部案例的低画质不阻断其他案例和后续渲染。
真正没有可用训练数据、权重或输出时仍报告失败，不能将空结果记为完成。
普通坏训练数据写入 `skipped_training_data`，不因少量跳过而把全部成片判为失败。
日志额外记录真实成功更新次数、梯度范数、输出头梯度、解码修正幅度及RGB对照。

smoke 将 train 上限设为48例、训练2步、监控2例，保留24卡和全部24个 dev 参考；
原生渲染采样减为4步。smoke 验证接口、梯度、分布式和恢复，不用于画质判断。

## 阶段和输出

启动脚本顺序为：

1. 复用既有脚本暂存冻结 RAE/Stream 权重。
2. `prepare`：并行建立训练/监控划分和 source/reference 表示、源对应提议缓存。
   输入与训练监督分开，记录身份和每例失败原因。
3. `train`：训练编辑传输模块，保存阶段和可恢复状态。
4. `predict`：用训练好的模块产生24例 dev 的可见编辑参考，并保留 RGB 提议。
5. `render`：同 source、anchor、允许区域、seed 和采样配置下输出三路：
   `native_reference`、`rgb_writeback`、`learned_rae_writeback`。
6. `summarize`：汇总覆盖率、失败、输入一致性、写回事件和媒体同步状态。

三路的意义分别是原生参考基线、无需 RAE 的直接传输、训练后的 RAE 传输。
更像 source 或颜色更淡不自动算提升；重点看相当编辑强度下的错位、闪烁、
背景污染、主体结构和目标外观保持。RGB对照用于避免把普通传输收益误记为RAE贡献。

输出仍双写到本次 `${OUTPUT_URL}/scale/edit_transport_learned_v1` 和持久化
`output/scale/edit_transport_learned_v1`。保留 `DI_throughput`及单位、每节点日志、
协调器心跳、阶段状态和NPU idle guard。guard计算不计入吞吐量。
各阶段分进程，避免 Stream/RAE 与14B同时驻留；训练时decoder仅保留所需反传图。

主要产物：

- `manifest.json` / `prepared.json`：配置与代码身份、分组、有效缓存和跳过记录。
- `inputs/<id>.pt` / `supervision/<id>.pt`：合法输入与训练监督独立保存。
- `checkpoints/step_*.pt` / `checkpoint_latest.pt`：阶段与最新模型、优化器、EMA、
  各rank RNG、监控状态；这是续训入口。
- `selected.pt`：依据 held-out train-group 监控损失选择的 **EMA**；用于预测。
  若没有完整监控结果，选择原因明确记录为最终EMA回退，不能称为监控最优。
- `monitor/step_*.json` / `training_summary.json`：监控覆盖、选择依据、收敛/跳过状态。
- `predictions/<id>/`：源视频、RGB提议、学习后的RAE参考和恢复payload。
- `render/{native_reference,rgb_writeback,learned_rae_writeback}/<id>/`：三路实际输出。
- `comparisons/<id>/native_rgb_learned.mp4`：同时间索引的 Native / RGB写回 / 学习RAE写回三列。
- `summary.json` / `media_sync.json` / `report.html`：完整覆盖、失败和同步信息。

缓存容量需要预留：每个训练视频的新增输入约27.15MB，独立监督约7.25MB，2048例
合计约70.4GB（约65.6GiB，未含序列化小额开销），dev额外约0.65GB。初次准备按
24卡分片，但训练的随机样本读取会使每个节点逐步缓存大部分训练数据，因此按
**每节点约70GB新增缓存**规划，而不是仅按三分之一。旧native bank、基础模型和
输出另外计算。训练缓存不保存稠密光流。准备耗时取决于Stream编码、源光流和OBS
双写速度，不能从CPU smoke或两步训练推算生产总时长。

## 单阶段恢复和路径覆盖

同一命名空间重复运行按已保存状态恢复。可单独运行：

```bash
LEARNED_STAGE=train bash scripts/scale/64_train_edit_transport.sh
LEARNED_STAGE=predict bash scripts/scale/64_train_edit_transport.sh
LEARNED_STAGE=render bash scripts/scale/64_train_edit_transport.sh
LEARNED_STAGE=summarize bash scripts/scale/64_train_edit_transport.sh
```

单阶段恢复保留之前的 effective config。恢复 smoke 时应显式指定：

```bash
LEARNED_NAMESPACE=edit_transport_learned_v1_smoke LEARNED_STAGE=render \
  bash scripts/scale/64_train_edit_transport.sh
```

更换训练设置或数据/参考协议时使用新 `LEARNED_NAMESPACE`，保留旧结果。
续训改变 selected 权重后，预测和受影响的成片会依内容身份重算；未改变的Native/
RGB可继续恢复。失败预测可重试，旧成功视频不能掩盖本次失败或旧权重。
原生/直接RGB仍可用时，训练或预测失败后脚本仍尝试出基线和最终汇总，最终退出码
保留失败信息。`render`、`summarize` 单阶段不暂存 RAE/Stream。

可用环境变量：

| 变量 | 用途 |
|---|---|
| `LEARNED_CONFIG` | 配置JSON路径 |
| `LEARNED_BANK_NAMESPACE` | 默认 `vace_edit_pilot_v1` |
| `LEARNED_BANK_LOCAL/PRIMARY/MIRROR` | 原生bank本地与两处读取路径 |
| `LEARNED_REFERENCE_NAMESPACE` | 默认 `edit_reference_video_v1` |
| `LEARNED_REFERENCE_LOCAL/PRIMARY/MIRROR` | 已完成Stage63参考输出的本地与两处读取路径 |
| `EDIT_RAE`、`EDIT_MODEL` | 已有RAE和原生VACE路径 |
| `LEARNED_TIMEOUT_SECONDS` | 每阶段协调超时，默认172800秒 |
| `EDIT_JOIN_TIMEOUT_SECONDS` | 等待节点就绪，默认3600秒 |

先看命令和路径，不加载模型：

```bash
LEARNED_STAGE=dry bash scripts/scale/64_train_edit_transport.sh
```

## 验证边界

本地共34项测试通过，包括真实 Bash 编排，覆盖阶段顺序、双读取/双输出参数、
训练失败后继续基线、smoke隔离、恢复配置、guard退出清理和带空格路径；以及
CPU模块、输入/监督隔离、训练、完整流程、缓存失效和媒体修复测试。
其中真实双进程Gloo测试验证参数更新同步、冻结decoder不变、
包含dropout/RNG/Adam/EMA的精确续训，以及单rank数据读取失败后集体跳过并继续更新。
提前中断时保存的无监控回退EMA会在恢复后刷新；已有有效监控最优权重则保持不变，
两种选择行为均有实际参数更新的回归测试。
任何本地测试都不替代真实14B、910B显存/HCCL或自然视频质量评估。
这份脚本不会在本地启动集群训练。
