# Stage55：StreamVGGT 轨迹动态 mask 小规模验证

2026-10-22。用户同意先验证相机补偿的动态 mask，再决定是否接入生成训练。
本阶段是离线诊断，没有修改生产 StreamVGGT、SceneAE、DiT、训练目标或缓存。
最新决定：默认复用以前训练的完整 StreamVGGT checkpoint，用户无需下载原版 VGGT。
真实 StreamVGGT/NPU 前向和 mask 质量仍须集群验证；本地逻辑测试通过不代表生成改善。

## 要回答的问题

1. 原始二维位移会把相机移动下的建筑标为动态；加入相机与深度解释后，误判能否减少？
2. mask 能否覆盖 AE 尚能重建、GEN 已重影的人物区域，且不主要覆盖遮挡/建筑边缘？
3. 只观察前4帧时，是否已有可用的运动证据？完整9帧结果只作为未来辅助诊断。

这一步不验证 dynamic embedding 能提高画质。若 mask 不可靠，不能据此修改 attention。
mask 即使可靠，也不能直接给出未来人物位置、速度和动作。
用户进一步明确：方法可以替换，优先真实画质，不必保留VGGT tracking方案。
若轨迹教师不可靠，下一步比较独立点跟踪/光流教师；若mask可靠而生成仍差，
转向运动对应、未来texture预测与解码敏感性验证，不继续无止境扫mask权重。

## 输入和教师

- 默认读取 `scene_street_motion_converge_v1_control/eval/step0006000/ema/` 的64个case、seed101。
- 使用普通四列MP4：RAW / AE / GENERATED / COPY。精确去掉32px标题栏，原518×518像素坐标不缩放。
- 输入是有损压缩预览，并非原始缓存RGB。局部像素指标不能与之前原始tensor指标直接横比。
- 源receipt校验只消费MP4和JSON；不为本诊断下载AE、DiT、latent `.pt` 或原始视频数据集。
- 默认教师使用已有完整 StreamVGGT 权重，生产因果 aggregator 配合官方相机、深度和跟踪头。
- 生产封装只接 aggregator；官方完整 StreamVGGT 有这些任务头。诊断单独组装模型，不改变生产 AE/DiT。
- 同源原始 backbone 特征能够被任务头读取，不代表压缩后的 C256 latent 也保留了同等可读性。
- 使用多帧窗口跟踪，不调用官方逐帧 `inference()`：该入口每次传入单帧，tracker 又固定输入首帧查询坐标，会得到退化的静止轨迹。
- 保持原始时间顺序，因果骨干不重新排列参考帧；跟踪头看整个输入窗口，不能把它的结果声称为逐帧在线预测。
- 必需的少量源码及许可证随仓库附带；集群不 clone、不访问 Hugging Face、不下载新模型。
- Stream heads/pose 固定上游 commit `05682bcb05683c4c71522e9cb618fc8ca349063c`，位于
  `third_party/streamvggt_trackmask/`，文件来源与 SHA 记录在 `UPSTREAM.json`。
  `utils/scene_trackmask_stream.py` 负责独立组装；深度头的位置编码用相同公式在 CPU
  计算 FP64 常量、转换 FP32 后送设备，避免在 Ascend 上创建 FP64 张量。
- 原版 VGGT 保留为显式 `vggt` 后端对照；它使用 `third_party/vggt_trackmask/`，并不是本轮启动依赖。
- 每个case分别执行完整9帧和仅观察前4帧两次前向；后者绝不从完整窗口结果裁切。
- aggregator默认BF16，camera/depth/track head显式FP32；无梯度，无优化器。真实NPU峰值仍待记录。
- 查询网格步长21像素、边距7像素，518输入上576个查询点；默认不分块，以保留tracker查询间交互。

全量所需 aggregator/camera/depth/track 权重逐键、逐 shape 检查；未使用的 point head 明确忽略。
缺失 head 不可用随机权重替代。只含生产 aggregator 的裁剪权重不够，需使用原始完整
`StreamVGGT/checkpoints.pth`。这属于模型兼容性检查，不是阻止训练的画质门槛。
模型种类、实际源码和 checkpoint 内容签名进入恢复身份，StreamVGGT 与原版结果不能混用。

## 动态证据的定义

用查询帧深度反投影3D点，再由相对相机投影到目标帧，与实际track比较。
反方向使用目标帧**独立预测的深度**，投影回查询帧；不复用前向预测z做无信息的代数往返。
正反向结论不一致、低可见性/置信度、深度边缘、可能遮挡、全局相机不可靠均保留UNKNOWN。
输出 STATIC=0、DYNAMIC=1、UNKNOWN=-1；软分数是启发式证据，不是经过校准的概率。

默认基础残差阈值为518分辨率下2像素，并加入5%深度不确定性对应的像素余量和背景median/MAD阈值。
至少两帧、至少50%有效比较帧支持独立运动才标为持续动态；短暂运动可能仍未知。
同时保存1/2/4像素和25%持续比例的小范围敏感性统计，不能只挑一个最漂亮阈值报告。
置信度和全局保护的精确定义由 `TrackMaskConfig` 记录在每份结果中。

已在受控测试复现：局部错误参考深度会让纯静态点出现>5像素单向残差。
加入独立反向证据后，该点变为UNKNOWN；平移/旋转相机下静态点、真正独立运动点仍通过对应测试。
这只证明逻辑能处理该构造，不代表真实场景误判率为零。双向相关深度错误仍可漏过保护。

稀疏点只支持半径10.5像素邻域；无点覆盖的地方不外推为静态。未知点或冲突的等距点保留未知。
这不是人物实例分割，查询帧后进入视野的对象也可能没有轨迹。窄小肢体和遮挡边界需要逐图检查。

## 集群启动

在既有ModelArts项目环境和代码更新后，每个节点执行同一个入口。默认自动识别节点数；
可以用1个节点，也可沿用3节点。每节点1个NPU0教师进程，按case编号分片，无HCCL集体通信。
其余已分配设备仍由既有idle guard覆盖，但这些工作不计入DI。这个小诊断不需要三节点训练预算。

沿用之前作业准备好的 StreamVGGT 权重和依赖，更新代码后直接运行：

```bash
bash scripts/scale/55_audit_scene_trackmask.sh
```

默认使用 `${VGGAE_REF_ROOT}/StreamVGGT/checkpoints.pth`（通常为
`/cache/yexiaoyu/vggae_ref/StreamVGGT/checkpoints.pth`）。若之前作业使用自定义位置，
继续设置 `STREAMVGGT_CKPT`；本地缺失时沿用 `STREAMVGGT_URL`、`STREAMVGGT_MIRROR_URL`
从已有主备存储准备。StreamVGGT 后端不会调用公网下载工具，即使遗留环境设置了下载开关。
所有节点使用同一份完整权重，恢复时不重复计算已保存的教师输出。

原版 VGGT 只是备用对照：显式选择 `SCENE_TRACKMASK_TEACHER_KIND=vggt`，另设新的
`SCENE_TRACKMASK_NAMESPACE`，并提供已经准备好的 `VGGT_TRACKMASK_CHECKPOINT` 及可选
`VGGT_TRACKMASK_CHECKPOINT_URL` / `VGGT_TRACKMASK_CHECKPOINT_MIRROR_URL`。集群始终设置
`VGGT_TRACKMASK_DOWNLOAD_OFFICIAL=0`；本轮无需准备这些原版模型文件。

没有为未知OBS位置编造地址。本次修改不解决当前结果桶访问故障；
默认输入/输出主备路径仍需可读写，也可通过下面的配置指向实际可用位置。
本机另打包了 `D:/workspace/VggAE_DataAudit/scene_trackmask_20261022/trackmask_input_stage54_control64.zip`
（约34MiB，仅64份预览、JSON和收据），可传至集群后解压，以解压根目录作为
`SCENE_TRACKMASK_SOURCE_LOCAL`，不需要重新传完整训练日志或原始数据集。

常用配置：

| 环境变量 | 默认 / 用途 |
|---|---|
| `SCENE_TRACKMASK_NAMESPACE` | `scene_trackmask_stream_stage54_v1`，新诊断结果命名空间 |
| `SCENE_TRACKMASK_TEACHER_KIND` | `streamvggt`；原版对照需显式 `vggt` |
| `SCENE_TRACKMASK_SOURCE_NAMESPACE` | `scene_street_motion_converge_v1_control` |
| `SCENE_TRACKMASK_SOURCE_LOCAL` | 本节点源run目录，可指向用户已准备的下载结果 |
| `SCENE_TRACKMASK_SOURCE_PRIMARY` / `..._MIRROR` | 源run根目录，主备读取 |
| `SCENE_TRACKMASK_CASES` | `0,1,...,63`；可先用`4,11,36,41,48`做短前向 |
| `SCENE_TRACKMASK_PRECISION` | `bfloat16`，只影响aggregator |
| `SCENE_TRACKMASK_QUERY_STRIDE` | `21`；不同网格需新命名空间 |
| `STREAMVGGT_CKPT` | 已有 `/cache/.../StreamVGGT/checkpoints.pth` |
| `STREAMVGGT_URL` / `STREAMVGGT_MIRROR_URL` | 已有完整权重的可选主备地址 |
| `VGGT_TRACKMASK_CODE` / `VGGT_TRACKMASK_CHECKPOINT` | 仅显式 `vggt` 后端使用 |

建议先沿用默认64例，不只挑选容易成功的人物案例。局部人物和强视差窗口已经包含在里面。
没有画质门槛阻止其他case完成。坏输入记录到case error并继续；错误不伪装为成功。
全局缺失/不兼容教师属于启动失败，不反复对64个case加载坏权重。

## 保存、恢复和查看

输出写入 ModelArts OUTPUT_URL，并镜像到owner的 `output/scale/scene_trackmask_stream_stage54_v1/`。
每case在教师推理后先保存并双写原始轨迹、深度、相机、置信度和校验收据，然后再计算mask/渲染。
渲染中断可以复用教师NPZ；缺失或损坏本地文件可从另一输出副本恢复。源码、配置、教师与输入签名进入恢复身份。
更改阈值、模型或代码时使用新namespace，避免把旧mask解释成新算法结果。

主要文件：

- `cases/case004/mask_review.mp4`：RAW / 原始位移点 / 刚性补偿点 / AE / GENERATED。
- `cases/case004/mask_review.jpg`：最后观察帧、第一未来帧、最后未来帧。
- `cases/case004/observed_prefix.jpg`：独立4帧前向，无真实未来输入。
- `cases/case004/metrics.json`：动态/静态/未知区域的相同支持集比较、覆盖率和阈值敏感性。
- `cases/case004/full_teacher.npz`、`prefix_teacher.npz`：可复算的教师中间结果，非模型checkpoint。
- `cases/case004/full_mask.npz`、`prefix_mask.npz`：正反向残差、标签、阈值、无效原因。
- `nodes/node000.json`、`node001.json`、`node002.json`：每节点完成/失败明细。
- `nodes/teacher_setup000.json`：所用后端、各头权重覆盖情况，或明确的初始化失败原因。
- `summary.json`：所有节点结果可读取后自动收集；只有部分结果时不假报完成。

颜色：红=动态候选、青=静态、黄=未知。画面上的线是刚性预测位置到实际track位置的残差。
原始位移列只用基础可见性/置信度筛选，不能借用补偿判定过滤掉建筑误判。
mask使用完整未来RGB的版本始终标为GT教师诊断；不能用这些mask作为正式生成条件。
本轮未实现dynamic embedding、attention屏蔽或运动预测训练。

`DI_throughput` = 本进程实际教师输入帧数 / 端到端墙钟时间；full和prefix分别计帧。
另记按全部已分配设备均摊的吞吐。IO、渲染、发布等待计入时间，守护计算与恢复命中不计帧。
下载和prepare-only的DI为0，并不是实际NPU推理吞吐为0。每case进度、30秒长阶段心跳、阶段NPZ均保留。

## 本机已做和待做

本机CPU已校验64份source MP4/JSON内容和帧序、去标题栏、三列对照图，全部成功，
结果在 `D:/workspace/VggAE_DataAudit/scene_trackmask_20261022/`。这些文件明确是prepare-only，
不是已算出的真实 StreamVGGT/VGGT mask。原始视频与完整教师权重未下载到本机。

本地测试覆盖相机平移/旋转、独立运动、局部深度错误、遮挡、置信度、尺度/世界坐标变换、
中文路径、双读双写、前缀输入隔离、中断恢复、真实MP4编码/解码和真实Bash启动/清理生命周期。
前一版59项诊断/下载测试、12项既有scene pipeline回归测试通过；本次 StreamVGGT 改造的验证记录另见下方。
`prepared_v2/summary.json` 明确记录64/64 prepare-only、0失败；本机对照图约31MiB。
`synthetic_geometry_results.json` 记录受控场景：相机平移/旋转时25/25静态点被原始位移规则误判，
补偿后0动态；加入一个独立运动点后检出1个；单点错误参考深度被标为未知。
教师使用假模型的集成测试只验证接口和流程。真实完整权重前向、Ascend算子和真实mask精度未验证。

2026-10-22 StreamVGGT 改造验证：73项诊断/下载回归测试、12项既有 scene pipeline
测试全部通过，Python编译和Bash语法检查通过。覆盖真实 Stream camera head 的因果隔离、
真实 tracker 多帧更新与单帧坐标锁定、带 Namespace 元数据的旧 checkpoint 加载、缺头拒绝、
默认 Stream 无公网下载、已有本地权重复用、主路径失败后镜像恢复、后端切换不能复用旧收据。
独立的小尺寸随机权重 DPT/TrackHead CPU 前向通过形状/有限值检查；它不是精度验证。
测试日志：`D:/workspace/VggAE_DataAudit/scene_trackmask_stream_20261022/tests_trackmask.txt`。
真实全尺寸模型的结构可在 meta tensors 上建立：aggregator1210、camera69、depth62、track394
个 state keys，共1735（不含未使用 point head）。这是预期结构，不是对集群现有文件的加载证明。
第一次集群启动将把真实覆盖情况写入 `nodes/teacher_setupNNN.json`；不需要先下载原版权重。

验收首先看静止建筑是否被大量标红、人物是否有足够有效覆盖、未知是否集中于合理位置，
再看相同区域AE与GEN之间的误差和可见形变。全图L1或动态点数量单独都不能判定成功。
若大面积UNKNOWN，先检查teacher相机/深度/跟踪证据，不自动放松阈值制造一个好看的mask。

相关方法背景：[StreamVGGT官方代码](https://github.com/wzzheng/StreamVGGT)、
[VGGT官方代码](https://github.com/facebookresearch/vggt)、
[PAGE-4D](https://arxiv.org/html/2510.17568v4)。后者的感知任务消融提示：
屏蔽动态跨帧attention可能改善相机估计却损害几何，不能直接移植成生成器的硬屏蔽规则。
