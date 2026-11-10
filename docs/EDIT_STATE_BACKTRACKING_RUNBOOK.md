# Stage67：带回溯的编辑状态优化

本轮读取 Stage65 已有的 RGB/RAE selected 权重，并复用 Stage64 源视频状态。冻结 VACE、LoRA、编辑接口和 AE，不新增权重训练。本轮只改变编辑状态求解器的步长与接受策略；24个案例、30步采样、三个反馈时刻、反馈强度以及五臂设置均与 Stage66 相同。

Stage66 的日志显示，很多联合优化候选不如初始状态，随后返回初始状态。Stage67 用回溯搜索尝试在同一目标下找到可接受的状态更新。是否改善语义、画质和 RAE 相对 RGB 的效果，仍需由本轮全部输出判定。求解目标下降本身不等于编辑成功。

## 启动

沿用已有集群启动与资源准备方式，三节点、每节点八张 910B 均执行：

```bash
FEEDBACK_STAGE=smoke bash scripts/scale/67_refine_vace_edit_state_backtracking.sh &&
FEEDBACK_STAGE=all bash scripts/scale/67_refine_vace_edit_state_backtracking.sh
```

不需要新的 bootstrap 或额外权重。默认端口30800；新输出命名空间 `edit_state_feedback_bt_v2`，smoke 自动使用 `edit_state_feedback_bt_v2_smoke`。smoke 仍读取完整 Stage65 checkpoint，只有案例数改为3、采样改为6步、求解改为一轮一步。

67是薄 wrapper，委托现有66脚本完成全流程，因此部分日志仍显示 `Stage66`。实际配置为绝对路径 `configs/edit_state_feedback_bt_v2.json`，输出命名空间可明确区分两轮。旧66脚本及 `edit_state_feedback_v1` 结果保留。

## 对照和优化设置

五臂顺序保持 `native / rgb_base / rae_base / rgb_joint / rae_joint`。基线满足原采样和checkpoint身份条件时复用 Stage65 的已完成输出；joint 两臂均使用新求解器，不能把只给RAE更优优化器产生的收益当成表示优势。

新增 solver 参数如下；其余参数保持 Stage66 原值：

| 参数 | 值 |
|---|---:|
| `optimizer` | `adaptive_backtracking` |
| `backtrack_max_trials` | 8 |
| `backtrack_shrink` | 0.5 |
| `step_grow` | 1.5 |
| `min_lr` | 0.000001 |
| `max_lr` | 0.03 |
| `armijo` | 0.0001 |

每帧作为一个更新块，接受检查覆盖其全部入边/出边约束。候选需满足全局目标严格下降和投影 Armijo 条件；每块最多尝试八次，非有限候选缩步重试。Adam 方向不是下降方向时重启动量，使用当前梯度重新寻找下降方向。全部拒绝时保留状态和动量，并缩小下一次起始步长；日志 `objective_improved=False` 明确表示未取得优化增益。

回溯会增加部分解码和目标评估，不能假定运行时间与旧66相同。按本轮记录的真实 `DI_throughput`、候选接受次数和逐案例耗时判断开销。无下降时保留有效状态、记录诊断，不能把“运行完成”或“反馈已写回”描述为优化成功。

## 恢复

```bash
# 继续未完成的完整实验生成
FEEDBACK_STAGE=render bash scripts/scale/67_refine_vace_edit_state_backtracking.sh

# 恢复 smoke：显式选择已存在的 smoke 命名空间，不再叠加 smoke 后缀
FEEDBACK_NAMESPACE=edit_state_feedback_bt_v2_smoke FEEDBACK_STAGE=render \
  bash scripts/scale/67_refine_vace_edit_state_backtracking.sh

# 只重新汇总/修复已有媒体副本
FEEDBACK_STAGE=summarize bash scripts/scale/67_refine_vace_edit_state_backtracking.sh
```

wrapper 的默认配置通过 `DEFAULT_FEEDBACK_CONFIG` 传给66，不设置用户级 `FEEDBACK_CONFIG`，因此恢复时可从已保存 manifest 找回 smoke 的有效配置。用户确需覆盖时，使用 `FEEDBACK_CONFIG=/path/custom.json`；与旧运行不一致的配置需要新的 `FEEDBACK_NAMESPACE`，避免混写不可配对的结果。

所有 Stage66 的覆盖项继续有效，包括 Stage64/65 三处输入根、`FEEDBACK_ARMS`、`EDIT_RAE`、`EDIT_MODEL`、`STREAMVGGT_CKPT`、协调时限和端口。三节点必须设置一致。

## 结果与保留策略

### smoke 在 `stage66_30` 结束并返回 1（2026-11-10）

更新：完整日志与产物已提供，确认实际为Stage68七臂，生成和报告已完成，
异常发生在最后发布/退出尾部。已限定汇总使用CPU后端，详见
[实测审阅与恢复命令](EDIT_STATE_CONTEXT_SMOKE_REVIEW_20261110.md)。下文保留
首次尾日志诊断和协调器遥测改进的背景，不能再将“产物是否完整”视为未知。

`stage66_30` 是汇总、对比视频制作与媒体副本核对阶段；只有 node0
执行汇总，另两节点的 0 不能证明各生成臂成功。`smoke && all` 中 smoke
非零会阻止 all，不能把 `&&` 改为 `;` 当作修复。Stage67/68 都委托66，
所以阶段名中的66正常。目录 `edit_state_context_sync_v1_smoke` 是68的
默认目录，与67默认不同；应查看 `launcher/node0/effective_config.json`
和 `manifest.json` 中的配置，不能仅按目录名判断实际实验。

ModelArts 的 CoreDumpUpload 记录只说明发生了 core 文件事件，尚不能把
它归因于汇总子进程、OOM、NPU idle guard 或权重加载。旧协调器会将负数
信号退出码归一化为1；新版额外保存 `child_pid`、`child_returncode`、
`child_signal`、`termination_reason` 和 `job.phase`，协议退出码保持原样。
汇总子进程开启 Python faulthandler 和无缓冲输出；这是诊断补强，不是
本次崩溃根因已修复的声明。Python 捕获到的缺失产物仍在 `summary.json`
与 `media_sync.json` 中报告，原生崩溃需看 `pipeline.log` 的堆栈。

只恢复用户这次已观察到的目录，三节点均执行下面命令；默认入口66从保存的
manifest 恢复有效配置，不需要判断当时用了67还是68：

```bash
env -u FEEDBACK_CONFIG FEEDBACK_NAMESPACE=edit_state_context_sync_v1_smoke \
  FEEDBACK_STAGE=summarize MASTER_PORT=30900 \
  bash scripts/scale/66_refine_vace_edit_state.sh
```

这不会重新采样或加载14B权重；它会读取已有结果张量、补齐媒体副本、重做
缺失的对比视频。若生成臂本身缺失，汇总仍应报告失败，按实际失败臂恢复
render 后再次汇总。保留此前日志；同目录恢复会追加 pipeline 日志并更新
阶段状态。诊断补丁不改生成代码签名，不要求新命名空间。

本次诊断补丁覆盖12项协调器测试与24项66/67/68启动、恢复测试，所有测试
均已有通过记录。首次批测有一项 Windows/Bash 的 `guard-stop` 标记缺失，
未改代码的单项复测通过；该现象不作为本次集群崩溃的解释。三个Python
文件编译、66 Bash语法和diff检查通过。未在910B上复现原始失败。

双读双写、NPU idle guard、失败案例记录、逐阶段继续、DI throughput、中间状态和反馈视频保存全部继承66。准备或某臂失败后仍执行独立臂和汇总，最终用非零退出标记实际失败，不新增画质门槛阻止后续阶段。

优先检查所有24例五臂最终视频、无损NPZ和反馈中间视频，而后比较旧66与新67的候选接受比例、最终输出和耗时。保持指标与编辑幅度仅作诊断，需要同时核对目标对象是否正确、编辑是否完成、结构及运动是否保持。仍属内部开发案例，不直接作为独立benchmark或SOTA证据。

`summary.json` 的 `feedback_health` 分开记录以下结果，避免再把注入已有参考计为新状态优化：

| 字段 | 含义 |
|---|---|
| `applied` | 实际完成 VACE 写回的反馈次数 |
| `solver_improved_events` | 求解器返回的有限目标值确实下降，之后仍可能因 VAE 编码失败回滚 |
| `committed_improved_events` | 目标下降且下游写回成功的反馈次数 |
| `committed_new_state_events` | 上述次数中，源网格编辑状态实际改变量非零的次数 |
| `accepted_block_steps` | 成功写回事件内部接受的逐帧候选步数；不等同于反馈次数 |
| `trial_evaluations` / `backtracks` | 所有反馈尝试的候选评估与缩步次数，包括随后回滚的尝试 |
| `cases_without_committed_improvement` | 没有成功提交目标改善的案例，仍保留视频和诊断 |

源网格状态改变量、门控后的 bridge 改变量和实际速度写回量各有独立诊断。源网格非零不能证明最终像素有正向收益。VAE 写回失败时完整恢复求解参数、Adam 动量、自适应步长和调用状态；正常全部拒绝仅保留缩小的下一步学习率。

## 本地验证（2026-11-07）

完整相关回归 **81项通过**（CPU，131秒），七个改动/新增 Python 文件编译通过，66/67真实 Bash 语法检查通过。覆盖实际 tiny VACE/UniPC 五臂流程、输入状态梯度、无编辑/零强度不变、全项双向图局部与全局目标差一致、投影约束、步长与动量回滚、完整双写/恢复及独立臂失败继续。

四轮近最优合成测试中，旧固定 Adam 的四个候选均差于初始状态；新求解器完成四次接受、11次候选评估和七次缩步：

| 求解域 | 固定 Adam 最终目标 | 自适应最终目标 | 相对下降 |
|---|---:|---:|---:|
| RGB | 0.000949990 | 0.000193997 | 79.58% |
| 非线性 RAE 测试 decoder | 0.000341451 | 0.000069244 | 79.72% |

这些数字验证缩步能处理所构造的过冲，不是生产 AE 或视频画质结果。910B/HCCL、实际显存、OBS 同步和最终画质仍需本轮集群验证。
