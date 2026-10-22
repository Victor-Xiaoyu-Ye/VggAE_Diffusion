# Stage55：VGGT 轨迹动态 mask 小规模验证

2026-10-22。用户同意先验证相机补偿的动态 mask，再决定是否接入生成训练。
本阶段是离线诊断，没有修改生产 StreamVGGT、SceneAE、DiT、训练目标或缓存。
真实 VGGT/NPU 前向和 mask 质量仍须集群验证；本地逻辑测试通过不代表生成改善。

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
- 教师为独立、完整的官方 VGGT-1B。生产StreamVGGT只保留aggregator，不能当作已有tracking输出。
- 官方模型不是本项目的生产表示；本诊断能验证外部VGGT教师是否提供有用信号，不能证明生产latent已含有同样的可读轨迹。
- 官方源代码固定commit并随仓库附带，位置 `third_party/vggt_trackmask/`，保留上游许可证。
- 官方checkpoint约5.03GB，只在集群 `/cache` 准备。本次未向本机下载大权重。
- 每个case分别执行完整9帧和仅观察前4帧两次前向；后者绝不从完整窗口结果裁切。
- aggregator默认BF16，camera/depth/track head显式FP32；无梯度，无优化器。真实NPU峰值仍待记录。
- 查询网格步长21像素、边距7像素，518输入上576个查询点；默认不分块，以保留tracker查询间交互。

具体来源和SHA见 `third_party/vggt_trackmask/UPSTREAM.md`。全量所需head逐键、逐shape加载；
缺失head不可用随机权重替代。公共官方权重与本地StreamVGGT/Omega权重不混用。

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

用户已明确集群不能访问公网。请在有网络的机器下载官方完整权重，再上传集群可访问的OBS，
或由作业准备阶段复制到各节点本地。同一份权重供所有节点使用，不需要分别下载不同模型。

唯一必需模型文件：[官方 model.safetensors](https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors)，
5,026,367,224字节，SHA256为 `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`。
模型Python代码已随项目附带；无需再下载`model.pt`或整个Hugging Face仓库。
集群使用本地/OBS模式，禁止开启公网下载：

```bash
VGGT_TRACKMASK_DOWNLOAD_OFFICIAL=0 \
VGGT_TRACKMASK_CHECKPOINT=/cache/yexiaoyu/vggae_ref/VGGT-1B/model.safetensors \
VGGT_TRACKMASK_CHECKPOINT_URL='<已有官方VGGT权重的OBS地址>' \
VGGT_TRACKMASK_CHECKPOINT_MIRROR_URL='<可选备用地址>' \
bash scripts/scale/55_audit_scene_trackmask.sh
```

没有为未知OBS位置编造地址。下载工具只准备教师，不解决当前结果桶访问故障；
默认输入/输出主备路径仍需可读写，也可通过下面的配置指向实际可用位置。
本机另打包了 `D:/workspace/VggAE_DataAudit/scene_trackmask_20261022/trackmask_input_stage54_control64.zip`
（约34MiB，仅64份预览、JSON和收据），可传至集群后解压，以解压根目录作为
`SCENE_TRACKMASK_SOURCE_LOCAL`，不需要重新传完整训练日志或原始数据集。

常用配置：

| 环境变量 | 默认 / 用途 |
|---|---|
| `SCENE_TRACKMASK_NAMESPACE` | `scene_trackmask_stage54_v1`，诊断结果命名空间 |
| `SCENE_TRACKMASK_SOURCE_NAMESPACE` | `scene_street_motion_converge_v1_control` |
| `SCENE_TRACKMASK_SOURCE_LOCAL` | 本节点源run目录，可指向用户已准备的下载结果 |
| `SCENE_TRACKMASK_SOURCE_PRIMARY` / `..._MIRROR` | 源run根目录，主备读取 |
| `SCENE_TRACKMASK_CASES` | `0,1,...,63`；可先用`4,11,36,41,48`做短前向 |
| `SCENE_TRACKMASK_PRECISION` | `bfloat16`，只影响aggregator |
| `SCENE_TRACKMASK_QUERY_STRIDE` | `21`；不同网格需新命名空间 |
| `VGGT_TRACKMASK_CODE` | 仓库中固定官方源码目录 |
| `VGGT_TRACKMASK_CHECKPOINT` | `/cache/.../VGGT-1B/model.safetensors` |

建议先沿用默认64例，不只挑选容易成功的人物案例。局部人物和强视差窗口已经包含在里面。
没有画质门槛阻止其他case完成。坏输入记录到case error并继续；错误不伪装为成功。
全局缺失/不兼容教师属于启动失败，不反复对64个case加载坏权重。

## 保存、恢复和查看

输出写入 ModelArts OUTPUT_URL，并镜像到owner的 `output/scale/scene_trackmask_stage54_v1/`。
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
不是已算出的真实VGGT mask。原始视频与5GB教师权重未下载到本机。

本地测试覆盖相机平移/旋转、独立运动、局部深度错误、遮挡、置信度、尺度/世界坐标变换、
中文路径、双读双写、前缀输入隔离、中断恢复、真实MP4编码/解码和真实Bash启动/清理生命周期。
最终59项新诊断/下载测试、12项既有scene pipeline回归测试通过；Python编译、Bash语法检查通过。
`prepared_v2/summary.json` 明确记录64/64 prepare-only、0失败；本机对照图约31MiB。
`synthetic_geometry_results.json` 记录受控场景：相机平移/旋转时25/25静态点被原始位移规则误判，
补偿后0动态；加入一个独立运动点后检出1个；单点错误参考深度被标为未知。
教师使用假模型的集成测试只验证接口和流程。真实完整权重前向、Ascend算子和真实mask精度未验证。

验收首先看静止建筑是否被大量标红、人物是否有足够有效覆盖、未知是否集中于合理位置，
再看相同区域AE与GEN之间的误差和可见形变。全图L1或动态点数量单独都不能判定成功。
若大面积UNKNOWN，先检查teacher相机/深度/跟踪证据，不自动放松阈值制造一个好看的mask。

相关方法背景：[VGGT官方代码](https://github.com/facebookresearch/vggt)、
[PAGE-4D](https://arxiv.org/html/2510.17568v4)。后者的感知任务消融提示：
屏蔽动态跨帧attention可能改善相机估计却损害几何，不能直接移植成生成器的硬屏蔽规则。
