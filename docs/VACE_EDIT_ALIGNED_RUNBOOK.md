# Stage61：验证 RAE 能否改善实际编辑（2026-11-04）

本轮已经实现，但尚未在 14B/910B 集群运行。目标是：给定源视频和文本，
改变指定对象的外观，同时保留其他对象、相机运动及原有动态。生成仍在
Wan 原生 VAE 空间；已有 Scene C256 RAE 是冻结的训练教师，推理不需要它。
不再把从零训练、替换 VAE 或长视频作为前置要求。

## 本轮的依据和可证伪问题

Stage60 已完整完成。24 个匹配 dev 样本，Native/SFT EMA/RAE EMA 的目标
PSNR 为 15.419/17.454/16.455。画面审阅支持 SFT 总体更稳；RAE 在帐篷案例
保护相邻墙面有收益，但蝴蝶、树林、河流等退化。人物案例少执行编辑，
不能把其更高 PSNR 当作成功。188 个视频的本地文件校验通过。

已确认的实现缺陷：teacher 将有效画面等比例放入 518 方形画布；旧 student
把整个长方形网格直接拉伸成 18×18。仅遮住黑边的 loss 没有修正坐标。
这解释一个确定的监督错误，不足以解释全部失败，更不保证修复后质量提升。

本轮回答三个问题：

1. 正确空间、时间索引下，RAE 监督是否改善主输出，而非只改善辅助头？
2. 普通区域加权 flow loss 能解释多少收益，RAE 是否提供额外价值？
3. 给所有方法相同的源视频区域后，定位错误和编辑/保持冲突各占多少？

## 实验矩阵

所有训练从相同原生 VACE 初始化；数据顺序、噪声、文本丢弃、LoRA、学习率
和优化步数匹配。不是从 Stage60 已训练的 SFT/RAE checkpoint 接着微调。

| arm | flow loss | RAE 监督 | 用途 |
|---|---|---|---|
| native | 不训练 | 无 | 原生模型，同样输入、采样参数 |
| sft | 普通全局 MSE | 无 | 任务微调基线 |
| rae | 普通全局 MSE | 修正对齐后的 latent + frozen-decoder RGB | 检验原始假设 |
| rae_detached | 普通全局 MSE | 只训练 readout，不回传主干 | 梯度路径负对照 |
| region_sft | 编辑/保持区域等权 | 无 | 必需的普通 loss 加权基线 |
| rae_balanced | 同 region_sft | 编辑/保持区域等权 RAE | 判断 RAE 的增量贡献 |

RAE head 和 LoRA 分别裁剪梯度。CPU 实际优化测试确认 detached arm 与 SFT
的全部 LoRA 更新完全一致；不是仅检查一个张量的 requires_grad。

**纯文本是主协议。** 额外的 4 个已审阅 dev 案例，使用人工指定的源视频
粗框，单独存储为 `*_source_region`。Native 和所有训练 arm 使用完全相同
的框。它们是诊断，不是无偏抽样，不是精细分割，不足以声称指令跟随更强。
这轮没有声称训练了通用点击交互能力；它利用 VACE 原有 mask 接口做推理
对照。关闭该附加诊断可设置 `EDIT_CONTROLS=''`，不改变纯文本训练。

## 数据与坐标

- 只读复用 Stage60 `vace_edit_pilot_v1` 的 native 和 teacher bank。
  不重新切分，不覆盖旧缓存，不下载全量 Señorita。
- 旧 bank 约 2048 train / 24 dev / 4 benchmark_dev；128 reserved test
  不训练、不选择 checkpoint。source 字节哈希分组不足以保证真实场景独立。
- 17 帧、8 FPS，面积上限 352×640、长边上限 768；保持长宽比，不放大原生
  RGB，pad16，输出裁回有效区域。官方 benchmark 自身 16 FPS 协议独立记录。
- 新 readout 保留 native 时空网格，明确反变换 pad16 与 teacher letterbox。
  native temporal slots 按因果 VAE 索引 0,4,8,…；teacher 索引 0,2,4,…16。
  线性插值按这些索引计算。它是索引对应，不声称每个 latent 只包含一个帧，
  或 token 的感受野是精确的像素点。
- RAE scale 的计算也排除黑边。辅助 decoder loss 的黑边使用 detached
  teacher token 作为共同上下文，避免把黑色 RGB 错当作全零 token。
  这些 teacher token 仅参与训练 loss，绝不送入 velocity 主路径或推理。
- `region_sft` 与 `rae_balanced` 使用同一份 source/target RGB 差异 support：
  3×3 平均、16/255 阈值、5×5 膨胀。它可能误包含压缩、错位或差 target，
  是变化代理，**不是语义、运动或几何标注**。
- 因果 VAE 区域支持使用 frame0、frames1…4、frames5…8 等组内并集，属于
  粗略影响区域。不能视为 VAE 感受野的精确逆映射。
- target-derived support 只用于训练损失和生成后的评估。推理粗框来自源视频
  人工标注；测试替换任意 target RGB 后，其实际 VACE context 保持不变。

## 启动与恢复

保留现有集群启动代码、依赖安装和 `vggae_ref` 位置。三个节点均在仓库执行：

```bash
EDIT_STAGE=smoke bash scripts/scale/61_train_vace_edit_aligned.sh &&
EDIT_STAGE=all bash scripts/scale/61_train_vace_edit_aligned.sh
```

新命名空间 `vace_edit_aligned_v2`，smoke 自动使用 `_smoke`。不是新 bootstrap，
61 是复用已有 60 启动流程的实验配置。使用 `&&` 避免 smoke 失败却继续 all。
smoke 是 3 步/4 采样步，RAE warmup=0、梯度诊断每步；仍复用旧完整 bank，
并非重建 48 条缓存。其配置中的样本数主要缩小预览请求。all 不受其 checkpoint
影响。每个 arm 都先走 2 步、落盘、换进程恢复，再完成剩余训练。

生产 AE 自动从既有 `scene_rae_c256_15day_v1/production/ae/best_joint.pt`
暂存。新一轮使用原生 VACE 权重，不依赖旧 arm 的优化器。已暂存的 native/
teacher 数据优先本地读取，否则从输出/持久化两处恢复并校验。

本轮 1000 步，batch=24，LoRA rank16，lr=5e-5，warmup50，accum1。
每 125 步保存不可变 checkpoint + latest + resume_pointer + optimizer/RNG/FP32 EMA。
每 250 步评估完整 24 dev 的 RAW/EMA；最终再做源框诊断。额外 2 步预览验证
只属于预检，不用于选择正式 checkpoint。每个独立 arm 失败后仍运行其他 arm。

仅补做某个已保存 checkpoint 的评估，不做优化器更新：

```bash
EDIT_STAGE=evaluate EDIT_ARMS='sft rae' \
EDIT_EVAL_CHECKPOINT=checkpoints/step_000250.pt \
bash scripts/scale/61_train_vace_edit_aligned.sh
```

仍需同一配置、命名空间和节点数。不能改配置后静默装载旧 optimizer。
恢复时用原命令；读取阶段 checkpoint、EMA、逻辑样本游标和 RNG。

## 双输出视频缺失修复

旧逻辑发现完成 receipt 后直接跳过。若 receipt/视频从持久化目录恢复，
本次 `OUTPUT_URL` 就可能缺视频。现在有两层处理：

1. 跳过生成之前，把恢复的完整视频和 receipt 补写到两处。
2. 全部独立 arm 结束后，node0 扫描两处 evaluation JSON，检查所有历史步
   receipt 指定的视频，按 size/SHA256 流式读回核验、修复，再写 receipt。
   临时失败重试；不重新加载模型，不遍历/复制大 checkpoint 或 latent bank。

单处暂时失败不毁掉训练；最终 `media_sync.json` 明确区分 `synced` 和
`incomplete`，后者令收尾退出非零，但不阻止已经排在前面的独立 arm。
后续只补同步即可，不需要重新训练。已有 Stage60 结果也能修：

```bash
EDIT_STAGE=sync EDIT_NAMESPACE=vace_edit_pilot_v1 \
bash scripts/scale/60_train_vace_edit_pilot.sh
```

新 Stage61 结果：

```bash
EDIT_STAGE=sync bash scripts/scale/61_train_vace_edit_aligned.sh
```

按原来三节点环境执行；协调器仅让 node0 做最终核验。若只拿到独立终端，
也可直接运行 `scripts/finalize_vace_edit.py --output <local-run> --root <job-output>
--root <persistent-run> --arms native sft rae`，不需要启动模型，确保 NODE_RANK=0。

输出保持：本次 `${SCALE_REMOTE_ROOT}/vace_edit_aligned_v2` 和持久化
`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/vace_edit_aligned_v2`。
旧 bank 的持久化位置是 owner 下 `cache_latents/vace_edit_pilot_v1`，不是新输出。

## 需要看什么结果

- `comparison_summary.json`：匹配样本、seed、prompt、空间尺寸和 control identity；
  不匹配/失败单列，不悄悄用更容易的交集宣布胜利。Native/SFT、SFT/RAE、
  SFT/detached、region_sft/rae_balanced 分别汇总。
- `<arm>/evaluation/s000250_ema.json` 等：完整 case 明细。纯文本与源框独立。
- `<arm>/samples/s001000/ema/<id>/source_target_output.mp4`：SOURCE / TARGET /
  对应 arm 输出，明确列标签。源框版本在 `ema_source_region`。
- `metrics.jsonl`：主 flow、aux、编辑/保持损失、有效 aux 比例、共享 hidden 上
  aux/flow 梯度 cosine 和范数比、DI_throughput、rank0 allocated 显存峰值。
  probe 不额外写参数梯度；不能把 hidden 的 cosine 当作整个模型梯度的夹角。
- 像素 PSNR 之外看 edit-target L1、keep-source L1、keep-target L1、实际/目标
  编辑幅度、keep 区域时序差分误差，以及 copy-source PSNR。变化代理指标
  不是独立几何准确率，低变化也可能是不执行指令。

当前估计全矩阵约 8–12 小时，依据旧版 SFT/RAE 实测约 94/84 全局帧每秒，
再加全 dev 预览和重复加载；不含不可预期的 OBS/权重下载等待。新 gradient
probe 的 NPU 成本尚未测量。原 RAE rank0 峰值 allocated 约 43.4 GiB；本轮
无第二套 14B 常驻，但不能以 CPU 测试保证 910B 峰值。先看 smoke 实测。
所有 phase 保留 DI 单位、心跳、双写、阶段保存和现有 8 卡 idle guard。

## 论文目标与近邻工作的边界

论文候选问题是：**外观应当改变的编辑任务中，如何让结构监督保留有用约束，
又不把颜色/材质和原物体状态一并锁住？** 已有 C256 可解码表示可能帮助
定位“该编辑的外观变化”和“破坏结构的误差”，但当前混合 RGB/Stream 表示
并没有被证明能解耦二者。修坐标、加 region weight 本身不是论文贡献。

- [Geometry Forcing](https://geometryforcing.github.io/) 已将视频中间特征与
  VGGT 对齐，包含方向及尺度目标。不能以“加 VGGT feature loss”为新颖点。
- [GAD](https://arxiv.org/abs/2610.18779) 已蒸馏几何注意力的跨帧对应，并随噪声
  选择学生层；静态表面与运动对象都涉及。不能泛称关系蒸馏首次解决动态。
- [VitronEdit](https://arxiv.org/abs/2610.27049) 已做图像/视频/3D 统一编辑与几何
  后训练，且其摘要报告只对 3D 样本几何监督效果最好。几何监督不必处处更强。
- [VACE](https://github.com/ali-vilab/VACE) 已支持文本、mask、视频等组合条件；
  源框输入不能成为我们的独占优势或更强纯文本能力证据。
- [Señorita](https://arxiv.org/abs/2502.06734) 是任务配对监督的来源。当前已经
  使用编辑专用数据，不声称无需编辑数据。

下一步由本轮实证选择，不再默认“再加一点 loss 就继续放大”：

| 观察 | 解释范围与后续 |
|---|---|
| aligned RAE > SFT，且 detached = SFT | 有主干监督增益，再拆 latent-only / decoder-RGB-only / 普通像素或感知监督，确认 RAE 必要性 |
| balanced RAE > 同权 region SFT，且编辑/保持同时改善 | 支持任务兼容表示约束方向；再构造可解释的结构/外观冲突处理，并做独立几何/运动评估 |
| 两种区域加权差不多 | 主要收益可由 loss 分配解释，不包装成 RAE 创新 |
| 源框让所有方法明显改善，RAE 没增量 | 当前瓶颈主要是对象定位/条件适配；不能证明 RAE 编辑收益 |
| RAE 仍退化或只降低改动幅度 | 停止堆全局 C256 回归；检验教师对合法编辑的敏感性、选层/残差或对应约束，必要时移除教师 |

达到投稿证据还至少需要：独立场景划分和去重；更多任务/视频；多个 seed 与
置信区间；同编辑执行程度下的非目标区域、独立 tracks/flow/depth/pose 指标；
盲评与原生任务强基线；等条件等算力消融。还要在非改色任务上看是否泛化，
并单独报告失败类别。不能以 24 个内部 dev 或 4 个选中粗框诊断代替论文评测。

本轮不需要用户新增下载。后续数据优先补**可靠的 source/edit prompt/edited
target 和独立评测划分**，人工确认目标对象、是否实际执行、是否误改背景。
源视频 object mask/track 可用于独立评测和区域输入协议；不要求所有训练数据
都有深度、位姿或分割。静态新视角/动态新相机暂不并入主任务，以免把未证明的
编辑质量扩展成另一个更难的生成问题。

## 本地验证边界

CPU 使用真实小尺寸 native VACE 骨架，不下载 14B；检验原生 forward、LoRA
反传、坐标 ramp（横/竖/方/原生 padding）、因果时间索引、冻结 decoder 梯度、
区域支持和零面积、detached/SFT 实际参数一致、替换 target 不改变推理 context、
两进程 Gloo + checkpointing + probe + 精确恢复、只评估 checkpoint 无 optimizer、
真实 Bash 编排/只读复用/失败 arm 隔离/只补同步，以及缺失损坏视频的双处修复。
NPU、HCCL、MoXing 实际服务和质量增益只能由下一次集群运行验证。
