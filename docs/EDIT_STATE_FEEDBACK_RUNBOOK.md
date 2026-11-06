# Stage66：冻结编辑模型上的联合状态反馈

本实验沿用已有三节点、每节点八张 910B 的启动方式。它读取 Stage65 已训练的 RGB/RAE selected 权重和 Stage64 源状态缓存，运行联合 decoder 校准与真实采样反馈；本轮不重新训练14B、AE或LoRA。最终效果仍需在集群实测，不能把本地合成测试当成质量证明。

## 启动

三台机器使用相同命令，位于已有项目目录下：

```bash
FEEDBACK_STAGE=smoke bash scripts/scale/66_refine_vace_edit_state.sh &&
FEEDBACK_STAGE=all bash scripts/scale/66_refine_vace_edit_state.sh
```

不需要新 bootstrap、外部安装或新增权重下载。脚本只按现有双读规则找回项目已有的冻结 AE/Stream 权重。smoke 使用三个案例、六步采样和一轮一步求解；输出自动追加 `_smoke`，但读取的是完整 Stage65 的 selected checkpoint，绝不改用两步 smoke checkpoint。主实验沿用配置中的24案例和30步采样。

## 对照与输出

默认按以下顺序执行，各臂独立进程：

1. `native`：原生 VACE＋相同合法参考。
2. `rgb_base`：冻结的 Stage65 RGB 编辑接口，不开启本轮联合反馈。
3. `rae_base`：冻结的 Stage65 RAE 编辑接口，不开启本轮联合反馈。
4. `rgb_joint`：共同输入及约束下的 RGB 状态反馈。
5. `rae_joint`：decoder 校准的 RAE 联合状态反馈。

旧基线与新增反馈都必须保留完整结果，包括失败案例。共享参考错误仍然计入失败，不能通过删除案例获得更好的均值。

## 本轮实际改变的内容

Stage63 已经有离线联合校准。本轮新增的是依赖**当前生成预测**的闭环：在实际采样噪声接近0.60、0.35、0.15时，解码当前干净预测，求解跨帧编辑状态，再更新状态分支与真实流速度。若多个请求映射到同一步，只执行一次。每次成功反馈后重新起步 UniPC 的多步历史，保留当前时间与噪声日程；失败和零编辑不改变历史。

求解使用九个已有源时刻，双向相邻边和锚帧远边各自计算前后向、光度可靠性。它是近似源图像对应约束，并不是新增可靠的3D重建监督。没有可靠编辑证据的位置保留当前生成预测，不强制复制源视频。奇数时刻通过完整17帧源对应传播修正。零修正不会引入518方形画布往返误差；实际视频保留原宽高比与有效区。

默认每次校准4轮，每轮逐帧反传，只留一个decoder反传图；信赖区域围绕非零初始编辑状态，使用Stage64训练源状态的通道尺度。读出为源视频加decoder编辑差值，不把整段视频替换为AE重建。校准参考与当前预测分别编码，相减后以0.35强度、每通道/时刻RMS上界0.25写回，排除独立锚帧prefix。

原有96通道VACE RGB条件保持共享的Stage65参考；本轮更新的是编辑状态分支和去噪变量，不能称RGB条件也在逐步刷新。RAE的状态更新按对应可靠性门控，未知区域保留旧状态，但后续网络的全局混合仍可能影响这些像素。默认RGB与RAE求解变量在同一种设备上，流计算和分离的图像缓存放CPU。RAE额外decoder计算要单独报告，不能把两者称为等算力。

## 结果怎么读

- `report.html`：每例同时间的五列视频；`comparisons/`另有更易观看的三列对照。
- `render/<arm>/<case>/generated_frames.npz`：17帧无损像素和FPS；MP4仅用于观看。
- `render/<arm>/<case>/feedback/step_*.pt`与MP4：每次成功校准的状态、参考、投影诊断、可视化，不用等整例采样结束。
- `summary.json`：全部案例均值、失败列表、`feedback_health`真实生效数/回退数/没有生效的案例。
- `media_sync.json`：最终视频和中间可视化的两根同步核查。

旧基线的源、锚帧、seed、selected权重、采样设置全部一致时，复用其已经生成的像素并重新发布媒体；旧结果丢失或不匹配则重新生成。最终视频完成不代表反馈生效，必须同时看`feedback_health`。keep/change等变化量不等价于指令正确性或画质。本轮先看RAE闭环是否超过RAE原版，再看相同操作的RGB闭环，避免把一般的锚帧加强误报为RAE独有收益。

本地默认输出：`${SCALE_ROOT}/edit_state_feedback_v1`。双写目的地沿用 `${SCALE_REMOTE_ROOT}/edit_state_feedback_v1` 与 `${SCALE_MIRROR_ROOT}/edit_state_feedback_v1`。每个节点在 `launcher/nodeN/` 保存完整日志、有效配置、coordinator 状态、NPU idle guard 与退出状态。

脚本在 CPU 准备与等待阶段也启动现有 idle guard；它不负责掩盖真实运行错误。各阶段失败时继续执行独立基线/其他臂与最终汇总，最终返回非零并保留成功结果。没有画质门槛阻止后续臂执行。

## 恢复与覆盖项

```bash
# 只准备合法输入与源对应
FEEDBACK_STAGE=prepare bash scripts/scale/66_refine_vace_edit_state.sh

# 恢复尚未完成的生成；runner 依据结果身份和 receipts 复用已完成案例
FEEDBACK_STAGE=render bash scripts/scale/66_refine_vace_edit_state.sh

# 只补指定臂，最终完整汇总仍会如实标记缺失臂
FEEDBACK_STAGE=render FEEDBACK_ARMS='rae_joint' bash scripts/scale/66_refine_vace_edit_state.sh

# 汇总与媒体副本核对
FEEDBACK_STAGE=summarize bash scripts/scale/66_refine_vace_edit_state.sh

# 仅读取启动设置，不加载模型或创建输出
FEEDBACK_STAGE=dry bash scripts/scale/66_refine_vace_edit_state.sh
```

`prepare/render/summarize` 会恢复已保存 manifest 的有效配置，防止中断后把 smoke 当成完整采样。显式 `FEEDBACK_CONFIG` 与原运行不一致时，应指定新的 `FEEDBACK_NAMESPACE`。恢复 smoke 时设置 `FEEDBACK_NAMESPACE=edit_state_feedback_v1_smoke`，并使用具体恢复阶段，不再次使用 `smoke`，避免重复后缀。

可覆盖输入目录：

| 环境变量 | 默认含义 |
|---|---|
| `FEEDBACK_STATE_NAMESPACE` | `vace_edit_state_v1`，Stage65 结果 |
| `FEEDBACK_STATE_LOCAL/PRIMARY/MIRROR` | Stage65 的本地、输出、持久化读取位置 |
| `FEEDBACK_TRANSPORT_NAMESPACE` | `edit_transport_learned_v1`，Stage64 源状态 |
| `FEEDBACK_TRANSPORT_LOCAL/PRIMARY/MIRROR` | Stage64 的三处读取位置 |
| `FEEDBACK_NAMESPACE` | 本次独立输出命名空间 |
| `FEEDBACK_CONFIG` | 本次配置 JSON |
| `FEEDBACK_ARMS` | 需要执行的生成臂，空格分隔 |
| `FEEDBACK_TIMEOUT_SECONDS` | 每阶段协调总时限，默认172800秒 |
| `EDIT_JOIN_TIMEOUT_SECONDS` | 等待各节点入场，默认3600秒 |
| `EDIT_MODEL`、`EDIT_RAE` | 沿用已有 VACE14B 与生产 Scene C256 权重 |

默认端口从30700分配，独立阶段使用不同 coordinator 端口，torchrun 使用各阶段端口加60。三个节点必须使用一致的配置、命名空间、权重和输入路径。

## 验证范围

2026-11-06本地统一回归58项通过：新增联合求解器、真实tiny VACE/UniPC反馈、五臂完整流程、真实Bash启动路径，以及既有Stage65状态/完整流程回归。覆盖零强度逐bit不变、两CFG读状态、真实输出改变、无可靠对应的bridge门控、VAE失败事务回滚、宽高比与prefix、旧基线payload缺失重新采样、中间/最终媒体双根缺失修复。Python编译、Bash语法、diff检查通过。测试中的生产AE/VAE权重由明确的小模型fixture代替；没有把这种通过当成真实编辑质量或910B通过。

launcher 测试以真实 Bash 执行脚本，用可结束的集群操作桩检查顺序、参数边界、失败继续、退出清理、smoke隔离、配置恢复、双读双写路径和固定端口。它不能验证实际 HCCL、OBS、910B峰值显存或输出画质。

实际算法、反馈日志和媒体完整性以本轮 runner 的实现及集群报告为准。论文必须区分共享 RGB 条件/LoRA 的收益与 RAE 独有收益；本轮新增反馈结果不自动等于 SOTA。
