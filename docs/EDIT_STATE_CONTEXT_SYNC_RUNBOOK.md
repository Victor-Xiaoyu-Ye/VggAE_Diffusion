# Stage68：同步校准后的编辑参考到 VACE 原生条件

首轮smoke的生成和报告已完成，但汇总进程非零退出，正式all未启动。
汇总已增加CPU后端隔离；现有产物可复用，详见
[实测审阅与恢复命令](EDIT_STATE_CONTEXT_SMOKE_REVIEW_20261110.md)。

2026-11-10。用户要求先交付最优先的三节点脚本，其余实验列待办。
本轮先完成最直接的主输出改造：使用现有 Stage65 训练权重，保留 Stage67 回溯，
把同一校准参考写入原生 VACE RGB 条件。没有重训 AE、DiT 或 LoRA。
这是24例开发集上的完整视频实验，不是已完成独立投稿主表，也不预设效果提升。

## 启动

在原有三节点，每台执行同一条命令，沿用既有项目/权重准备：

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
FEEDBACK_STAGE=smoke bash scripts/scale/68_refine_vace_edit_state_context.sh &&
FEEDBACK_STAGE=all bash scripts/scale/68_refine_vace_edit_state_context.sh
```

仍是 3 × 8 Ascend910B，MASTER_PORT 默认30900。没有新增下载、pip安装或bootstrap。
读取完整 `vace_edit_state_v1` 的 RGB/RAE selected checkpoint，以及
`edit_transport_learned_v1` 的源状态和已有合法参考。smoke读完整训练权重，
只把案例缩为3、采样缩为6步，输出为独立 `_smoke` namespace。

主运行 namespace：`edit_state_context_sync_v1`。与 Stage65/66/67 全部隔离。
主配置：`configs/edit_state_context_sync_v1.json`，24例、17帧、8FPS、30步采样。
画布宽高比、pad/裁回、源时刻、锚帧、seed、求解器、反馈强度全部沿用Stage67。

## 一次跑完的比较

| 顺序 | arm | 用途 |
|---|---|---|
| 1 | native | 原生 VACE + 共同锚帧，原source-reactive条件 |
| 2 | rgb_base | Stage65 RGB，无在线反馈 |
| 3 | rae_base | Stage65 RAE，无在线反馈 |
| 4 | rae_joint | Stage67式RAE反馈，原生RGB条件仍固定 |
| 5 | rae_sync | 新方法：同一校准参考同步更新原生RGB条件 |
| 6 | rgb_joint | Stage67式RGB反馈，固定原生条件 |
| 7 | rgb_sync | RGB使用同样同步操作，验证收益是否是通用条件更新 |

先输出 RAE 旧/新主对比，再跑 RGB 对照；一个 arm 失败不阻止剩余 arm 和汇总。
符合身份的历史 native/rgb_base/rae_base 可复用，缺失则重新生成；四个反馈版本
在本次命名空间中实际运行。旧 Stage67 joint 不被当作新版本结果静默复用。

## 实际改动与边界

旧流程已改变 state bridge 和velocity，但下一步 VACE 仍读取最初 RGBproposal。
新版在成功反馈时，用校准后参考按原始mask编码 **reactive video** 部分，替换
96通道条件中的第16:32通道、去掉独立prefix的时序槽。源inactive编码、64个mask
通道和参考prefix逐bit保留，不重算，不把干净latent误当成masked reactive条件。

当前步仍用既有有界clean-residual校正velocity；新RGB条件从下一对CFGforward起
生效，和更新后的bridge共同参与后续采样。成功提交后重置UniPC多步历史，保留原
时间表。RAE的可靠性混合及velocity的alpha/RMS限制仍存在；不能声称非线性RAE
桥状态、RGB条件和最终像素数学上完全等价。本实验检验的是移除旧参考固定的限制。

条件在每次sample调用内复制，避免污染磁盘缓存或下一次same-seed比较。
solver、VAE编码、finite/grid检查全部完成后才提交state/context。失败时回滚
solver与两种可变输入，沿用原velocity继续；alpha=0和no-edit保持精确no-op。
没有新增画质门槛阻止后续实验。参考本身指错对象仍可能失败，这里未暗中加入GT或
新的语义分割器；不存在新未来标签输入。

## 读结果与恢复

- `comparisons/<case>/rae_context_sync.mp4`：RAE base / 固定条件反馈 / 同步条件反馈。
- `comparisons/<case>/rgb_context_sync.mp4`：RGB同样三列。
- `comparisons/<case>/matched_context_sync.mp4`：Native / RGB同步 / RAE同步。
- `render/<arm>/<case>/generated_frames.npz`：最终无损17帧；MP4用于观看。
- `render/<arm>/<case>/feedback/step_*.pt`：每次成功反馈的state、projection、
  live reference；sync版本还包含实际native_context。
- `summary.json`：全部臂/案例、失败、旧指标、solver进展，以及新增
  `context_requested_events/context_updated_events/context_delta_rms_max`。
- `media_sync.json`：最终和中间视频的双根校验/缺副本修复。

默认主输出位置：

```text
/cache/yexiaoyu/vggae_runtime/outputs/scale/edit_state_context_sync_v1/
${OUTPUT_URL}/scale/edit_state_context_sync_v1/
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/edit_state_context_sync_v1/
```

具体根路径以启动日志配置为准。DI_throughput保留明确单位；中间状态每次反馈后
落盘，最终视频逐例保存。沿用日志双写与idle guard，空闲占卡工作不计入DI。
从中断恢复时重跑同一命令，完成案例由身份/receipt恢复；未完成的单例从相同seed
重新采样，不从无效的半截UniPC历史继续。

```bash
# 只恢复生成；使用已保存的有效配置
FEEDBACK_STAGE=render bash scripts/scale/68_refine_vace_edit_state_context.sh

# 只补一个失败臂，汇总仍会如实标记其他缺失项
FEEDBACK_STAGE=render FEEDBACK_ARMS=rae_sync bash scripts/scale/68_refine_vace_edit_state_context.sh

# 只做汇总与双写修复
FEEDBACK_STAGE=summarize bash scripts/scale/68_refine_vace_edit_state_context.sh
```

预计在权重/缓存已就绪后约1–2小时完成smoke+all，属于预算估计，实际以每例DI和
阶段耗时为准；新增reactive VAE编码会增加耗时，不能直接沿用Stage67的22分钟。
旧RAE peak allocated约40GiB，新版910B峰值/效果要看本次smoke与all。

## 验证与后续

本地测试覆盖真实tiny VACE/UniPC两条CFG读到新context、最终输出改变、旧缓存不变、
官方masked context重编码等价、prefix/source/mask逐bit保持、求解/编码失败回滚、
七臂完整流程、真实Bash启动、smoke/恢复、失败继续及两根缺副本修复。
测试使用明确的小codec/AE fixture，不能替代生产14B/910B/HCCL/OBS与画质验证。
101项互不重复的相关CPU/Bash测试通过，7个Python文件编译、3个Bash语法检查通过。

其余节点待办见 `docs/SUBMISSION_TODO_20261110.md`。本轮完成后再决定是否训练
同步条件适配，而不是未验证条件契约便投入6000步；原48/72小时排期据此调整。
