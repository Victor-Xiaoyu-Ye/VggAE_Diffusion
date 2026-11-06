# Stage65：在 VACE 去噪中训练编辑状态条件

本轮目标是让完整编辑状态实际参与每一步去噪，优先获得相同参考输入下优于原生
VACE的成片。上一轮只优化中间参考、再做两次晚期写回；本轮训练VACE的LoRA和
状态投影接口。RAE与RGB必须保持公平输入，最终视频用于判断收益。

## 启动

使用现有三节点、每节点8张910B和原有代码/权重暂存方式。三台节点在仓库目录
执行相同命令，无需新增bootstrap、联网安装或下载其他模型：

```bash
STATE_STAGE=smoke bash scripts/scale/65_train_vace_edit_state.sh &&
STATE_STAGE=all bash scripts/scale/65_train_vace_edit_state.sh
```

正式命名空间`vace_edit_state_v1`；smoke自动使用`vace_edit_state_v1_smoke`。
默认主端口30600，各阶段协调与torchrun端口分别偏移。

只读复用三个已完成实验：

- Stage60 `vace_edit_pilot_v1`：原生编辑bank、训练对和文本embedding。
- Stage64 `edit_transport_learned_v1`：源表示、源光流RGB提议和既有train/monitor分组。
- Stage63 `edit_reference_video_v1`：24例dev的合法模型/用户首帧参考。

未来edited target仅作训练监督；不能用来产生dev参考、编辑条件、mask或状态。
训练/monitor中paired target首帧是明确声明的参考编辑协议，不是纯文本编辑。
Stage64的小残差不能直接冒充完整编辑状态；其主编辑在RGB旁路里，本轮重新编码
完整合法RGB参考。RGB控制不会偷偷使用RAE校准后的RGB作为输入。

## 三路比较和实际接入点

| 分支 | RGB条件 | 状态和可训练部分 |
|---|---|---|
| Native | 原生source-reactive＋相同首帧reference | 原生冻结权重 |
| RGB | source-inactive＋共享RGB提议-reactive＋相同reference | 原始RGB/source-edit patch状态桥＋rank16 LoRA |
| RAE | 与RGB分支完全相同的RGB条件 | source/完整编辑RAE状态差的桥＋同rank16 LoRA |

RGB和RAE两臂的原生96channel条件均保留source的非编辑部分，允许区域使用同一
源光流RGB提议。不能直接将整段source替换为reference后编码，否则非编辑部分也
会丢失原source。Native使用其原生source-reactive条件，收到相同source/anchor/
mask/seed；这不是“Native只给文本、我们额外给图片”的对照。

新增状态桥在`vace_patch_embedding`之后加入VACE条件token，再经过VACE控制块
和主DiT。RGB和RAE臂的采样器必须调用该wrapper，正负CFG分支都传入同一状态，
避免沿用旧采样接口时绕过bridge。默认不依赖Stage64的晚期两次写回。

RAE输入为源C256、完整参考与源之间的C256编辑差、源对应置信度和支持区域，
共514通道。RGB控制使用同位置16×16的source RGB和edit RGB patch、置信度和
支持区域，共1538通道。两者桥宽128、结构相同，均零初始化输出；RGB输入投影
参数稍多，必须报告实际参数量，不能声称严格等参，也不靠补零伪造等宽。

九个RAE时刻与VACE五个video latent时刻显式对齐；合法首帧reference的prefix
单独处理。训练clean前缀来自该anchor，video flow loss排除prefix；采样后先
去除prefix latent，再进行causal VAE解码。空间使用实际pad/letterbox坐标，
保持横竖比；不能将有效区域直接拉伸到方形。

RAE decoder校准只对合法RGB参考的编辑差进行，默认4步；decoder权重冻结，
保留可恢复的最佳状态及前后误差记录。校准不读取未来target、不改共享RGB参考。
它的作用是让传给生成器的状态具有可检验的解码含义，不代表校准本身必然提高
最终视频质量。

## 训练预算和显存

- 两个学习分支均从相同原生VACE开始，rank16 LoRA，bridge宽128；不续接某个
  只属于一臂的旧SFT权重。StreamVGGT、RAE和VACE原始参数冻结。
- 最大1500步，最少500步后检查停止；每250步以固定监控样本/噪声/四个sigma
  `.15/.35/.6/.85`评估。使用24个原train内留出样本，不依据24个dev成片选权重。
- 每125步保存阶段checkpoint，含参数增量、优化器、EMA和各rank RNG。
  每500步保存最多8个dev预览；最终仍生成全部24个dev三路视频。
- 学习率5e-5、warmup50、EMA.99、梯度裁剪1；text/state dropout各.1，
  identity训练比例.1，flow编辑区域权重比例.5。
- RGB/RAE共享最大预算和采样/噪声约定，各自依据监控选择EMA与结束步数。
  早停可能导致实际更新次数不同，应如实记录，不能自动称为等训练计算量。

原有Stage61相近rank16训练曾记录约39GiB的NPU allocated峰值，但本轮新增
reference时间槽和状态桥，不能直接沿用这个数值作为显存保证。使用BF16原生
权重、块级activation checkpoint、只给LoRA/bridge建优化器；训练时不驻留
StreamVGGT/RAE，也不重复保存14B原始权重。smoke先执行真实小步训练并记录
峰值与梯度；本地CPU测试不能替代实际910B/HCCL/显存验证。

离线缓存、训练、渲染拆成进程：`states`用Stream/RAE；`cache`用VAE；
`train`用VACE；每个arm单独启动，避免多个14B或teacher同时占卡。
此前新增缓存的80GB估算偏低，现按下表重新核算。以352×640 canvas、17帧、1920个train、
24个monitor和24个dev为例；GB用十进制，MiB用二进制：

| 每例保存内容 | 约MiB |
|---|---:|
| source/reference uint8 RGB与单帧anchor | 22.56 |
| mask/confidence FP16 | 14.61 |
| RAE状态514×6×22×40 FP16 | 5.18 |
| 两组96×6×44×80 BF16原生条件 | 7.73 |
| 两组16×5×44×80 BF16训练监督 | 1.07 |
| 小计，不含文本embedding/序列化 | 51.15 |

dev保存编辑/原生条件而没有训练监督，空间差异很小。1968例小计约105.5GB；
三组文本embedding的长度会再增加空间，若每组均达512×4096 BF16，最多再约
24.8GB。实际按**约110–135GB/节点新增热缓存**估计，建议该部分单独预留
150GB并根据首次实测修正，不将其称为所有视频比例下的上界。其他canvas面积
会近似线性改变RGB、mask、状态与VAE缓存大小。

Stage60/64旧缓存、完整模型、两臂checkpoint、预览和最终视频均另算；smoke
还有独立命名空间。训练检查点保存参数增量、EMA与两组Adam状态，每份约为
16×可训练参数量字节，保留多个阶段快照会累积，不能只预留一个LoRA文件大小。
若每臂约6400万可训练参数、保存到1500步，12个阶段快照加latest约需每臂
13.3GB，两臂约26.6GB；selected、smoke和文件开销再另加，以实际参数量为准。
RGB1538通道patch状态在线构建，不额外保存全量patch缓存。随机训练会逐步把
大部分样本恢复到每个节点，不能按三节点平均分摊本地热缓存空间。

[Stage61实测](VACE_EDIT_ALIGNED_RESULTS_20261105.md)中，各独立1000步arm完整
训练阶段约76.6–81.4分钟，末尾DI约90.7–96.9全局目标帧/秒，包含当时IO、采样
和保存。按相同效率外推本轮两臂各1500步，合计训练约3.8–4.1小时，仅作为
排期参照；本轮多一个reference latent槽，状态处理、监控/预览频率及缓存规模
也不同，不能承诺该时长。此前约7.19小时是Stage61整个矩阵，不能直接套到本轮。
额外安排离线states/cache、三路30步最终生成、同步与smoke时间；首次启动后
以各阶段DI、已完成case数和本轮NPU峰值更新剩余时间。

smoke保留24卡、全部24个dev；只使用原train中的最多48例、2步训练、2个监控
case、4步生成，禁用训练中预览。它检查真实接口和执行，不用于判断画质。

## 阶段、故障处理和保存

`all`/`smoke`依次运行：

1. 既有teacher暂存助手。
2. `states`：构造相同RGB参考、完整RAE编辑状态、校准与对齐记录。
3. `cache`：缓存原生VAE条件与独立训练监督。
4. `train --arm rgb`，随后`train --arm rae`。
5. `render --arm native`、`rgb`、`rae`，分别启动生成进程。
6. `summarize`：比较、覆盖率、失败和双目录媒体核验。

普通坏数据记录后跳过；某臂失败不阻止独立的其他臂、基线和汇总。脚本保留非零
退出码，缺失模型/训练/视频不会伪报完成。每阶段均有协调心跳和独立日志。
不把低画质或某项指标未达阈值作为阻断其他臂的门槛。

输出双写到本次`${OUTPUT_URL}/scale/vace_edit_state_v1`与持久化
`output/scale/vace_edit_state_v1`。缓存、checkpoint、媒体保留双读恢复与双写；
沿用NPU idle guard和日志publisher。`DI_throughput`须标明单位与全局/单rank口径，
guard计算不计入真实吞吐。

主要训练状态位于：

- `arms/{rgb,rae}/checkpoint_latest.pt`
- `arms/{rgb,rae}/checkpoints/step_*.pt`
- `arms/{rgb,rae}/selected.pt`
- `arms/{rgb,rae}/training_summary.json`

观看和复核入口位于同一输出根目录：

- `report.html`：24例三路对照总览及运行/覆盖摘要。
- `comparisons/{case_id}/native_rgb_rae.mp4`：同时间轴的Native/RGB/RAE并排视频。
- `render/{native,rgb,rae}/{case_id}/`：各分支的生成记录、receipt和视频。
- 上述每例目录的`generated_frames.npz`：裁掉padding后的uint8生成帧及fps，
  不经过MP4有损编码，供后续定量核查；浮点生成结果另存于`result.pt`。
- `previews/{rgb,rae}/s{step}/`：训练中EMA预览；预览不用于dev择checkpoint。

最终比较与覆盖以`summary.json`、媒体同步记录及实际三路视频为准。恢复时使用
最新训练状态继续参数/优化器/EMA/RNG；最终生成使用所选EMA，不能把最后一步
权重和监控最优权重混称。checkpoint变化后的旧生成缓存必须按权重身份失效。

## 恢复与覆盖路径

单阶段恢复优先双读本次`manifest.json`中的配置；新节点没有launcher日志时也能
恢复smoke配置，不会误用1500步正式配置。若manifest暂不存在则沿用本地effective
config；两者冲突，或显式`STATE_CONFIG`与已保存配置不同，会明确报错，避免
静默改变已运行实验。三台节点仍需执行相同阶段：

```bash
STATE_STAGE=states bash scripts/scale/65_train_vace_edit_state.sh
STATE_STAGE=cache bash scripts/scale/65_train_vace_edit_state.sh
STATE_STAGE=train bash scripts/scale/65_train_vace_edit_state.sh
STATE_ARMS=rae STATE_STAGE=train bash scripts/scale/65_train_vace_edit_state.sh
STATE_STAGE=render bash scripts/scale/65_train_vace_edit_state.sh
STATE_STAGE=summarize bash scripts/scale/65_train_vace_edit_state.sh
```

`STATE_ARMS`仅选择训练恢复的分支，默认`rgb rae`；最终render仍尝试全部三路。
恢复smoke应指定完整命名空间，例如：

```bash
STATE_NAMESPACE=vace_edit_state_v1_smoke STATE_STAGE=render \
  bash scripts/scale/65_train_vace_edit_state.sh
```

变更数据、方法或训练配置使用新`STATE_NAMESPACE`。只读检查路径可执行
`STATE_STAGE=dry`，不会暂存权重、启动guard或训练。

| 环境变量 | 作用 |
|---|---|
| `STATE_CONFIG` | 配置JSON |
| `STATE_NAMESPACE` | 新实验输出名称 |
| `STATE_BANK_NAMESPACE` | 原Stage60 bank名称 |
| `STATE_BANK_LOCAL/PRIMARY/MIRROR` | bank三处读取路径 |
| `STATE_TRANSPORT_NAMESPACE` | 默认Stage64缓存名称 |
| `STATE_TRANSPORT_LOCAL/PRIMARY/MIRROR` | Stage64缓存三处读取路径 |
| `STATE_REFERENCE_NAMESPACE` | 默认Stage63参考名称 |
| `STATE_REFERENCE_LOCAL/PRIMARY/MIRROR` | 合法参考三处读取路径 |
| `STATE_ARMS` | 训练阶段要恢复的rgb/rae臂，空格分隔 |
| `STATE_TIMEOUT_SECONDS` | 每阶段协调超时，默认172800秒 |
| `EDIT_JOIN_TIMEOUT_SECONDS` | 节点就绪等待，默认3600秒 |
| `EDIT_MODEL`、`EDIT_RAE` | 已有模型/RAE本地路径 |

## 结论边界

先看相同编辑条件下是否有更好的实际成片，再看固定集合数值；source保持改善
不能以不执行编辑为代价。RGB对照回答状态表示是否提供额外价值；仅比Native
好不能证明全部收益来自RAE。模型参考改错对象仍属于端到端失败，当前接口
不会自动纠正错误指令定位。几何优势和论文贡献均需实验建立。

本地共59项测试通过：数据状态9项、新运行时10项、完整流程4项、双卡Gloo2项、
旧VACE路径回归23项、Bash启动11项。完整流程使用真实tiny VACE及UniPC、实际
梯度更新和视频写入；生产Stream/RAE/VAE权重由明确的小型codec fixture替代，
不代表生产编码器质量验证。覆盖未来target隔离、空间/时间prefix、FP16缓存、
两臂真实参数更新、三路输入身份、权重变更后重采样、坏数据继续，以及NPZ写入
中断后从已有结果修复。最终视频和训练预览均验证双目录缺失媒体恢复。

本地launcher测试执行真实Bash，覆盖阶段顺序、双读双写参数、独立失败继续、
smoke隔离、单阶段和单臂恢复、带空格路径与guard清理；还验证新节点从镜像
manifest恢复smoke配置、冲突报错和匹配的显式配置，11项通过。
`tests/test_vace_edit_state_ddp.py`的2项真实双进程Gloo测试通过：64维、2层原生
VACE配合生产514/1538通道状态输入；RAE直跑4步与2+2恢复的可训练权重、EMA、
Adam和各rank RNG严格相同，冻结骨干权重不变；RGB单rank读取失败后全部rank
共同跳过，再完成后续3次更新，失败步DI为0。真实14B/910B/HCCL、OBS与视频
质量均须集群验证。
