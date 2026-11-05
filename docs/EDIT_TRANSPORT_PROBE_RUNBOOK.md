# Stage62：固定 AE 上的编辑传输机制实验

实现日期：2026-11-04；生产运行审核：2026-11-05。smoke/all 已在三节点 910B
完整完成，结果见 [实测报告](EDIT_TRANSPORT_PROBE_RESULTS_20261105.md)。H1 校准
优于直接 token 传输；H2 当前 RAE 胜过 VAE 对照但落后于 RGB 补全，不能宣称
RAE 独有编辑收益已经建立。数学设计见
`docs/DECODER_CALIBRATED_EDIT_TRANSPORT_20261104.md`。

## 依赖与启动

**不依赖 Stage61 的训练结果。** 读取 Stage60 `vace_edit_pilot_v1` 的24个
dev源视频缓存，以及已经确定的生产 Scene C256 AE 和 StreamVGGT 权重。
Wan只加载现有 `Wan2.1_VAE.pth` 做一个图像codec对照；不加载14B DiT、文本模型、
SAM2或新增模型，不联网安装/下载。预留128个test不参与本实验。

用户将另起新作业接在上一轮后面：保留原有三节点启动、代码和refs暂存方式。
先把 OBS 代码包更新到本次提交；三个节点执行相同命令：

```bash
TRANSPORT_STAGE=smoke bash scripts/scale/62_audit_edit_transport.sh &&
TRANSPORT_STAGE=all bash scripts/scale/62_audit_edit_transport.sh
```

不需要重跑 Stage61。旧作业若尚在运行，本实验也可以在另一个独立资源作业
运行；同一组三节点按上述用户安排顺序运行。默认3节点×8张NPU，24个case
按global rank分配，源编码、RAE求解和VAE求解都并行。

命名空间：正式 `edit_transport_probe_v1`，smoke自动为
`edit_transport_probe_v1_smoke`。每个phase单独进程释放模型/显存。
固定顺序：暂存教师 → source编码 → RAE/RGB实验 → VAE实验 → 汇总与双目录修复。
RAE phase失败仍尝试VAE和汇总；不会因质量差于基线而阻止后续phase。

## 这轮具体做什么

不是自然视频生成或文本编辑benchmark。每个真实源视频只取第一帧，保持比例
letterbox到518，构造9个已知二维平移/旋转/缩放状态，模拟可控的已知对应。
这些状态与真实3D相机、自然人体运动不同。全部图和摘要明确标注synthetic。
不读取数据集edited target来构造source或推理条件。

源窗口通过实际StreamVGGT和固定C256 AE编码。默认目标时刻为第4/8状态，
编辑强度0.5/1.0，编辑区域为预定义软椭圆，颜色增量[.02,-.03,.10]。
这是空间编辑探针，不是语义物体mask或已验证的albedo操作。
decoder使用FP32单帧反传，不形成完整Jacobian，也不依赖NPU JVP或grid_sample
backward。当前生产decoder是无temporal blocks配置，本实验不测试新增时序decoder。

H1检查传输与解码是否交换：

- edited anchor仅改变已知第一帧，计算其编码增量；希望传输的是该增量实际
  解码出的RGB变化。这个目标是decoder定义的，所以H1不跨codec宣称优越。
- 对比不编辑、RGB oracle、直接token warp、token效果的最优单标量校准、目标源状态的decoder校准、
  使用warped-anchor近似响应的控制。所有方法以同一编辑强度/对应比较。
- RGB oracle在这个完整正确对应实验里是上限参考，不是公平击败的对象。

H2检查不完整观测下的读写补全：

- ground truth改用**独立指定的解析RGB编辑**，与任何codec的重建无关。
- 观测覆盖1.0/0.25，观测噪声std=0/0.005。缺失像素真实值只进入最后评估，
  不进入优化、初始化或best-state选择。已知编辑区域保留一个小seed，所有方法相同。
- 对比不编辑、稀疏RGB、源图边缘感知的RGB补全/去噪、RAE逆写、原生Wan VAE逆写。
  这里噪声是effect观测噪声，不冒充真实错误光流。
- RGB对照有归一化粗初始化和边缘感知harmonic补全，观测为软约束，可去噪，
  不用“缺失区填黑且不补全”的弱基线代表RGB方法。
- Wan VAE用T=1图像模式，518画布居中pad到528后裁回；单帧尺度与输入RGB一致。
  RAE encoder看9个源状态、两者表示容量/重建质量不同，必须记录这些差异，
  不能把全部收益归于VGGT几何。

这轮没有DINO-RAE权重对照、自然对应模型、native14B去噪写回或完整编辑前端。
它们在`summary.json/not_evaluated`明确列出。不是把尚未实现的整篇方法当成已跑通。

## 规模、恢复与输出

正式配置每卡负责一个源片段，约1600次decoder优化反传，全部24卡并行。
完整任务数：RAE/RGB phase中RAE960、RGB1152；VAE384，共2496个小cell。
本次正式 launcher 阶段实测约15.81分钟，smoke约7.65分钟，不含外层安装/下载；
OBS状况与冷启动会影响时间。`decoder frame evaluations/s/npu`不是生成视频FPS。
smoke保留24个源片段以覆盖每张卡，缩小为1个目标时刻/强度/观测设置、2次更新。

每10次更新保存输入delta、Adam状态、best状态及逻辑step，两处同步；每cell
完成保存全分辨率计算的指标、最终delta和128px effect预览、内容校验receipt。
518px评价与128px保存预览分别标注；不把预览当指标计算来源。特定固定设置
额外保存全分辨率四列PNG。源码、配置、bank、AE/encoder权重哈希组成恢复身份；
改变求解设置或核心代码时使用新命名空间，不静默混用旧结果。

中断后用原命令继续，已完成cell恢复后也会补写到新输出目录。无需从头求解。
`DI_throughput`分别标注source编码帧/秒/卡和decoder评估帧/秒/卡，避免当作
生成视频吞吐。每5步回显/发布进度，所有phase保留8卡idle guard及协调器心跳。

输出：

- 本次 `${OUTPUT_URL}/scale/edit_transport_probe_v1`
- 持久化 owner下 `output/scale/edit_transport_probe_v1`
- 原生源bank继续只读owner下 `cache_latents/vace_edit_pilot_v1`

主要文件：

- `summary.json`：覆盖、失败、同源clip配对差值和描述性bootstrap区间。
- `report.html`：离线查看四列诊断PNG；原始source、观测信息、已知效果、实际读写结果。
- `metrics.jsonl`：完整per-cell指标；不是loss下降就宣布效果改善。
- `source_previews/<id>.mp4`：9个受控2D源状态、AE重建、解析编辑。第三列不是生成模型输出。
- `cells/.../optimizer.pt`：中断恢复状态；`result.pt`有可重放delta及effect预览。
- `media_sync.json`：逐receipt payload检查两处SHA256/大小，失败标记incomplete。

只补汇总/双写修复，不加载模型：

```bash
TRANSPORT_STAGE=summarize bash scripts/scale/62_audit_edit_transport.sh
```

## 怎样判断是否值得继续

先看`heldout_edit_l1`、`keep_leakage_l1`和`effect_gain`，再看总体误差。
不编辑基线的gain为0，不能靠减小所有改动取得一致性胜利。
配对按同clip/frame/强度/观测/噪声，区间按source clip重采样，不把多个cell
当作独立视频。只有一次探针seed，不作论文显著性声称。

H1改善仅支持decoder校准这个机制；H2比边缘感知RGB/VAE没有增量，就不支持
RAE在这个设置的额外作用。负面结果也正常完成，不用质量门槛中止另一组。
下一步自然动态/遮挡、强文本前端和真实去噪写回应由这轮结果决定。

## 本地检查边界

已用真实小尺寸Wan VAE验证pad/crop和decoder输入反传；验证已冻结参数不累积
梯度、完整逆问题精确中断恢复、任意替换隐藏target不改变求解状态、完整缓存/
两codec/汇总/媒体修复流程；双Gloo进程验证分片和汇总；真实Bash入口验证独立
顺序、smoke隔离、失败phase后继续、所有设备idle guard和只汇总路径。
本次生产910B完整运行及双目录核验记录均成功；本地仍缺9张VAE PNG和`.pt`
payload，不能从本地下载声称已验证张量恢复或峰值显存。完整数值可用于审阅，
不需要重跑计算。机制效果和优化预算的边界见实测报告。
