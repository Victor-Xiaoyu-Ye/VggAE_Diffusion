# Scene 生成器原始权重 / 历史 EMA 重采样

2026-10-09。入口 `scripts/scale/52_audit_scene_flow_weights.sh`。针对 stage51 低精度 EMA 更新停滞，先比较现有原始模型和原样保存的旧 EMA；不训练、不更新 EMA、不重新编码数据。见 [问题证据](SCENE_RAE_RESULT_REVIEW_20261008.md)。

## 启动

更新 `ascend-910b` 后，在 ModelArts 现有 PyTorch/Ascend 环境使用以下启动命令。三个节点运行相同命令，保留平台自动提供的节点编号和主节点地址：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/52_audit_scene_flow_weights.sh
```

支持每节点8卡、一个或多个节点；现有三节点配置可直接用。没有固定只让rank0计算，case与权重分配给全部rank。少量case与某个world的组合可能有空闲rank，保活覆盖等待时间。

默认输入是 `scene_rae_c256_15day_v1/production`；输出是全新 `scene_rae_raw_ema_audit_v1`。无需设置stage51的 `SCENE_STAGE`，也不要重跑51号 `all` 来期待修复旧视频。

默认做完整固定验证集27例，保留原来的索引和 `seed=101+index`，Euler32、CFG1。以下每个checkpoint分别采样 `online` 与 `legacy_ema`，共 **4×2×27=216** 个结果：

| 输出目录 | 原训练 checkpoint |
|---|---|
| `image_final` | `image/checkpoint_final.pt`，原始步数16,394 |
| `video_s9185` | `video/weights_step0009185.pt` |
| `video_s71609` | `video/weights_step0071609.pt` |
| `video_final` | `video/checkpoint_final.pt`，原始步数137,104 |

实际内部step会记录，周期快照还会核对文件名步数。`online`取checkpoint的`model`；`legacy_ema`取`ema`，精确保留旧BF16数值载入FP32模型，**不把它当成已经修好的EMA**。两组推理均使用原来的BF16 autocast。32步先用于隔离权重因素，64步/Heun应另起输出namespace再比较。

如仅想先验证最后视频checkpoint两个案例的执行链：

```bash
SCENE_AUDIT_NAMESPACE=scene_rae_raw_ema_smoke_v1 \
SCENE_AUDIT_PROFILE=smoke SCENE_AUDIT_MAX_CASES=2 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/52_audit_scene_flow_weights.sh
```

smoke没有训练、没有画质门槛；完成后用默认命令运行完整矩阵。默认standard本身也可直接运行，不强制先跑smoke。

## 数据、设备与恢复

- 只读冻结AE、cache/eval.pt、归一化统计以及四份生成器checkpoint。验证缓存已有RGB、文字、相机；无需StreamVGGT、旧R7、T5、原视频或训练cache重建。
- 每节点由local rank0下载并校验一次大型checkpoint，其余卡从节点共享文件mmap读取；同时仅有一套生成器+AE在每卡，不恢复optimizer、不额外分配EMA模型。CPU/NPU实际峰值与吞吐由此次集群执行记录。
- 大文件存到新审计目录私有 `_inputs/nodeN`；各卡释放后清理当前生成器临时文件。冻结AE和eval留作缓存。原训练目录和OBS对象不修改、不删除。私有inputs不会被上传为输出。
- 对源checkpoint、AE、eval的SHA/identity以及采样配置固定contract。原始global case index不因卡数或筛选改变；增加/减少节点仍可按每例receipt恢复。改steps/method/profile/案例范围须用新namespace。
- 每例立即保存预览、生成latent、采样过程统计、指标及hash receipt，并双写平台路径与owner OBS。失败案例记录error后继续其余案例；最终partial不冒充完成，重启会重试失败/缺失案例。画质差不会触发停止。
- 中断后直接重跑相同命令。已完成case必须通过receipt+内容校验才跳过；损坏本地/主副本可从镜像恢复。成功完成也可重复运行，修复副本而不重复采样。
- 日志每30秒显示staging等待，采样每8步显示进度，每例输出字面 `DI_throughput: ... tokens/s/npu`。计数为去噪网络处理的序列token×调用次数，不含通道、不乘world；分母为单worker本次作业墙钟时间，含staging/IO，不含保活计算量。不是旧训练DI的同口径效率对比。
- 沿用作业作用域NPU idle guard，默认覆盖下载/同步等低利用率时段；退出等待guard终态、排空日志，再同步退出记录。它不构成对平台回收策略的绝对保证。

## 输出与阅读

owner输出：

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/scene_rae_raw_ema_audit_v1/
```

`summary.json` 每个checkpoint结束更新；`complete.json` 保存最终状态和全部案例指标。优先配对看：

```text
image_final/online/caseNNN/preview.mp4
image_final/legacy_ema/caseNNN/preview.mp4
video_final/online/caseNNN/preview.mp4
video_final/legacy_ema/caseNNN/preview.mp4
```

每例三栏为RAW/AE/GENERATED。视频第0帧是参考latent解码，评价只统计未来帧；图像任务不读取参考首帧内容作生成条件。`generated.pt`保存积分终点的归一化latent及过程统计；`metrics.json`标注原始checkpoint SHA/step、原始存储dtype、seed和作用范围。`complete.json`为该例提交receipt。`launcher/node*/pipeline.log`和保活日志仍双写。

这次首先回答原始权重是否明显好于旧EMA，以及视频中期/终点的差别。无条件图像与一张特定GT的L1不适合作生成质量裁决；同时检查可辨认内容、结构与运动。

## 本地验证边界

8项CPU测试验证与旧采样的逐值一致性、无首帧泄漏、固定seed、分卡分配、双镜像恢复和源文件不变；其中实际启动两个Gloo进程，验证节点leader单次暂存、各rank共享读取与完整汇总。shell语法与stub launcher验证退出/日志同步。实际910B、HCCL、Linux大文件mmap和真实权重生成质量，需要这次集群执行验证。
