# Stage68 smoke：产物完成后的退出异常

本地证据目录：`D:/中转站/VGGAE/scale/edit_state_context_sync_v1_smoke`。
这份审阅不把进程异常标记为成功，也不把 smoke 指标作为完整质量实验。

## 已确认的运行状态

- 实际 manifest 是 Stage68 七臂配置，三例、17帧、8fps、六步采样；
  不是 Stage67 默认五臂。三节点使用同一运行身份。
- 准备和全部七臂协调器均返回0；各臂均有三个 `completed` 案例。
- `summary.json` 为 `completed`，`failures=[]`、`media_failures=[]`。
- `media_sync.json` 记录48份媒体 receipt 的核对完成，无失败。
- 本地12个对比MP4的大小和SHA256均与三份 comparison receipt一致；
  全部完整解码为17帧、8fps。下载目录没有 `render/` 原始结果，所以
  不声称独立复核了所有原始张量/单臂视频；本次也未重新读取OBS对象。
- manifest 的六个生成源文件SHA256与当前仓库逐个一致。
- node0 的最终 publication status 记录两处输出成功、无错误，时间为
  `1794298043.2443666`，即北京时间16:07:23.244。
- 用户平台日志在16:07:24报告 core 文件事件；node0 的阶段状态在
  16:07:27.105记录 `stage66_30 exit=1`。另外两节点返回0。

因此，生成和报告产物已经完成，故障窗口位于最后报告发布/进程收尾。
日志没有该阶段的 Python traceback，也没有原始信号退出码/core堆栈，
不能进一步宣称已确定为 torch_npu、MoXing、ffmpeg、OOM 或某个析构函数。
summary 的 completed 不等于 shell 成功；`&&` 阻止 all 的行为正确。

## 修复范围与恢复

已确认一个不必要的运行库依赖：纯CPU汇总子进程仍加载 Ascend runtime。
对应 `plog-49807_20261110160654563.log` 有 runtime/HCCL初始化记录；代码
路径 `ArtifactStore -> utils.training -> utils.device` 在模块导入时无条件
尝试导入 `torch_npu`。PyTorch本身还会默认加载设备后端插件。

修复将 `VGGAE_CPU_ONLY=1` 和 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 仅限定在
阶段30的协调器及其汇总子进程。`utils.device` 在此显式模式下不导入NPU
后端、不探测加速卡。准备、采样及独立 idle guard 保留原后端。上游开关
核对自 [PyTorch 2.7.1 源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/__init__.py#L2507-L2562)。

这是移除无用NPU依赖的针对性修复，尚不是对原生core根因的证明。
保留前一补丁的信号/PID记录和 faulthandler，不使用 `os._exit(0)`，
不根据旧 summary 强制将失败改成成功。生成器、配置和生成代码签名不变，
已有 smoke 产物可以恢复。

更新仓库后，三个节点使用相同命令，先仅恢复汇总，通过后进入正式七臂：

```bash
env -u FEEDBACK_CONFIG -u FEEDBACK_ARMS \
  FEEDBACK_NAMESPACE=edit_state_context_sync_v1_smoke \
  FEEDBACK_STAGE=summarize MASTER_PORT=30900 \
  bash scripts/scale/68_refine_vace_edit_state_context.sh &&
env -u FEEDBACK_CONFIG -u FEEDBACK_ARMS \
  FEEDBACK_NAMESPACE=edit_state_context_sync_v1 \
  FEEDBACK_STAGE=all MASTER_PORT=30900 \
  bash scripts/scale/68_refine_vace_edit_state_context.sh
```

summary恢复不会重新采样或加载14B权重。若仍异常，保留此次新增的原始
`child_returncode`、`child_signal` 和堆栈；不能自动跳过非零退出。

## 本地验证

34项测试通过：三个全新Python进程测试使用可被PyTorch真实发现的模拟
设备插件，验证自动加载确实能触发、正常设备导入保持不变，以及CPU汇总
导入整套runner并双写/读取CPU张量时不会导入NPU。另有七项tiny模型实际
采样/媒体恢复测试和24项真实Bash启动/恢复测试，确认CPU限制只在汇总，
prepare/render仍用原环境。五个Python编译、Bash语法和diff检查通过。
这些测试不等于已在910B复现并排除原始core，汇总恢复仍是集群验证步骤。

## 本轮能支持的实验结论

RGB/RAE sync 各6/6事件完成原生context更新，均无fallback；四个反馈臂
各6/6事件改善内部求解目标并提交新状态。这支持“新条件写入路径运行了”，
不支持“画质提高”。六步采样的更新在step4和step5，末次context更新已无
后续采样forward，进一步限制了此smoke的质量解释。

| 保持区域变化（越小表示改动更少） | fixed context | synced context | 相对变化 |
|---|---:|---:|---:|
| RGB | 0.03059937 | 0.03065399 | +0.1785% |
| RAE | 0.03554679 | 0.03560898 | +0.1749% |

这三例低步数结果没有显示保持指标改善。保持指标也不是语义遵循/画质，
不据此选优或声称RAE超过RGB。下一步仍是24例、30步正式配置的完整对照。
