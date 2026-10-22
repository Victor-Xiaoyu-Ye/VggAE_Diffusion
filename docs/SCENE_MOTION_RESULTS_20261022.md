# Stage54 固定窗口 motion 对照结果

2026-10-22 检查用户手动下载至 `D:/中转站/` 的 control/motion 两组结果。全程使用本地文件，没有访问 OBS，没有读取或下载模型权重。结论是：训练和评估完整结束，但本轮 latent 变化加权没有带来生成画质提升。更有价值的新证据来自分支替换诊断：当前主要 RGB 偏差能通过真实未来 texture latent 替换修复，geo 替换的净收益很小。该证据尚不能判定 geo 分支是否必要。

## 作业与产物完整性

| 项目 | control | motion |
|---|---:|---:|
| 资源 | 3×8 Ascend 910B | 3×8 Ascend 910B |
| 更新数 | 6000 | 6000 |
| 总耗时 | 7.986 小时 | 7.982 小时 |
| 停止原因 | optimization plateau | optimization plateau |
| 源权重评估 | 128/128 | 128/128 |
| 1000–5000 每次监控 | 32/32 | 32/32 |
| 最终普通评估 | 256/256 | 256/256 |
| 节点退出 | 3/3 exit 0 | 3/3 exit 0 |
| 最后100条 DI 均值 | 1429.96 | 1417.55 |

DI 单位为 future latent tokens/s/NPU，含实际 IO、保存和评估等待，守护工作不计入。日志内最大 allocated 显存为25.397 GiB；这不是设备总占用或 reserved 峰值。

48份 rank 训练日志的 step 序列一致且递增，每份601条（step1、10、…、6000）。全部记录数值有限，无裁剪、bad shard、OOM、Traceback 或发布失败记录。48个设备守护正常清理，busy/pulse 最大记录间隔约675秒；没有空卡两小时回收的证据。这是日志证据，不是连续硬件利用率采样。

1088份普通 case 收据均 completed，3306个非 `.pt` 附件的大小/SHA全部匹配，1130个MP4全部以FFmpeg完整解码通过。每组565个MP4包括544个普通预览及21个GT辅助诊断。本地没有 `.pt` 本体，因此没有验证checkpoint内部内容、所有远端副本或真实恢复执行。

本轮的架构、数据、源权重与 [runbook](SCENE_MOTION_CONVERGENCE_RUNBOOK.md) 一致：冻结生产C256 AE与原统计，831,305,344参数SceneFlow，原32768街景窗口，观察F0–F3、生成F4–F8。两组各自从stage53 step31069 online开始，重新初始化AdamW/FP32 EMA。motion没有接着control训练，无相机/文本条件或新增标注。

## 同步数、同场景、同噪声的比较

匹配键包含 case、场景ID、frame_ids、seed、noise_seed、history/future及采样步数。每case先平均两个seed，再汇总64个case；这些不是独立训练重复。

| 未来5帧 RAW L1，越低越好 | 结果 |
|---|---:|
| 起点31069 online | .08940048 |
| control6000 online | .08942967 |
| control6000 EMA | **.08890248** |
| motion6000 online | .08982384 |
| motion6000 EMA | .08926629 |
| AE重建参照 | .04380931 |
| COPY最后输入帧 | .10496468 |

motion相对control，online/EMA分别差0.441%/0.409%，仅在18/64、16/64个case有更低L1。EMA场景配对bootstrap差值95%区间为[+.00022355,+.00050994]；该区间只描述验证场景变化，不包括训练重复的不确定性。

control EMA相对源online均值改善0.557%，motion EMA改善0.150%；两者相对源的场景bootstrap区间均跨0。stage53最终EMA原记录约.08913，因此本轮也没有建立比旧参考明显更好的画质。control EMA在52/64个case优于COPY，motion为51/64。不能把小数第四位的变化写成质量突破。

| 最终EMA的公共指标 | control | motion |
|---|---:|---:|
| 全局监督去噪MSE | .04590068 | .04589527 |
| 高变化区域监督去噪MSE | .05820380 | .05803629 |
| 未来相邻RGB变化幅度误差 | .01688210 | .01641854 |
| 观察→未来边界变化幅度误差 | .01351470 | .01354606 |

motion的高变化区域MSE改善约0.288%，相邻RGB变化幅度误差改善约2.75%，但这不是光流或人物动作准确率。该加权可以改变变化幅度，同时不修复人物结构。最高GT帧差四分位的16case也没有体现RGB误差优势。不能比较两组各自的weighted training loss来判胜。

## 停止是否合理

两组1000步建立EMA探针参考，2000步的global/change两指标都改善，3000、4000、5000、6000四次未达到相对0.2%的改善，stale依次1/2/3/4，故以 `optimization_plateau_requires_visual_review` 停止。

| 固定前16 case、seed101 | control EMA RAW L1 | motion EMA RAW L1 |
|---|---:|---:|
| 1000 | .09480212 | .09528799 |
| 2000 | .09287302 | .09327254 |
| 3000 | .09340496 | .09405938 |
| 4000 | .09342602 | .09391574 |
| 5000 | .09429120 | .09475517 |
| 6000 | .09408510 | .09453534 |

相同前16例的online比较也在所有保存步数上由control略优。2000是这16例当前较好的快照，但没有完整64例结果，不选作已证明的全局最佳。

最后LR仍约3.0535e-6，并非cosine已经走完。停止只说明既定6次监督探针触发平台，不证明所有学习率、数据覆盖或更长训练都无效，也不等于视频质量达标。末次summary由16例扩到64例，必须筛回前16例比较；不能把最终global .0459直接和2000步的约.04337比较后认定突然退化。

## 实际画面

查看了两组最终seed101全部64例的末帧、7个重点案例的时序/局部对照及004/011/041的motion EMA分支诊断；完整解码不等于逐个直接观看所有1130段视频。

建筑、道路及较简单的运动继续保持原基线水平，control与motion大多数画面接近。case004/011的AE仍能保留分开的腿部，GEN从首个未来帧出现边缘融合、重影，随后加重。motion没有修复这些问题。case007近景头发/遮挡、036树干视差、048人物边界的失败也保留。case041在AE就存在明显曝光异常，不能算生成器独有错误。

现有可观看短预测基线得到保留，没有之前全图崩坏；这不意味着动态人物已经合理，也不证明任意固定窗口生成已经解决。

## 分支诊断是本轮最有用的新信息

下表只含预先选择的7个诊断case、seed101、未来F4–F8，不能和上面的64case均值混用。GT指真实未来帧编码的缓存latent，不是深度或位姿真值；hybrid使用未来GT，不能当生成结果。

| 七例平均 RAW L1 | AE | GEN | GT geo + pred tex | pred geo + GT tex |
|---|---:|---:|---:|---:|
| control online | .072219 | .123321 | .123082 | .072019 |
| control EMA | .072219 | .122837 | .122626 | .072095 |
| motion online | .072219 | .123682 | .123432 | .071992 |
| motion EMA | .072219 | .123141 | .122966 | .072073 |

28个case/权重/实验组合全部是texture替换收益更大；其140个未来帧都获得改善。去掉AE曝光异常case041后，control EMA六例为AE=.034554、GEN=.086161、GTgeo=.086086、GTtex=.034495，现象不由单个异常例造成。源31069已经有相同现象，因此不是本轮加权引起。

control EMA七例GEN的L1从F4的.095998增长到F8的.146651，而AE约.074443→.070653；GTtexture替换始终贴近AE。这提示许多新增误差发生在未来texture预测或两支配合中，单独强化AE局部重建不一定消除它们。

不能据此说geo完全无用。去掉标签、只比较未来帧的有损MP4，GEN↔GTgeo的平均像素差约.01237，GEN↔GTtex约.07962；geo确实改变像素，只是对RAW L1的净收益很小。当前仍有“geo预测已经足够准确”“decoder局部对geo低敏感”“两支信息重叠/替换造成分布变化”等解释。反归一化分支MSE量纲不同，不能直接比较大小来确定原因。

代码上，SceneRAE decoder直接拼接geo与texture；teacher恢复头和关系约束都看完整z，没有强制VGGT来源半支对RGB不可替代。texture是逐帧RGB空间编码，能够承载位置和形状，名字不代表它只存颜色纹理。geo/texture也不是静态/动态分解。故本轮没有证明“几何分支预测差导致画质差”，也没有证明“VGGT对画质无贡献”。

## 下一步安排建议

保留stage53 step31069 FP32 EMA作连续研究参考，stage54 control6000 EMA作配套候选，不把motion设置升为默认。当前结果不支持继续扫描同类变化权重系数或只延长这版训练来期待明显提升。

先在冻结现有权重、固定case/噪声的条件下补分支必要性诊断：对clean GT与实际sampled latent分别做geo均值/零替换、跨场景shuffle、按通道标准差校准的扰动曲线，并对texture做对称对照。保存精确decoder张量差异和局部轮廓，区分离分布破坏与真实贡献。无需重新训练或额外人工标签。

若geo必要而其预测足够好，后续重点是texture未来建模、RGB敏感方向与decoder对生成误差的稳健性，再考虑低噪声感知约束或几何条件下的texture建模。若geo不必要，则须先调整表达和重建信息流，确保几何贡献可验证，再扩散或加位姿/WASD。两种都是条件性研究方向，本轮没有实现新结构、mask AE或控制训练。

## 查看位置与本地证据

本轮普通视频：

```text
D:/中转站/scene_street_motion_converge_v1_control/eval/step0006000/ema/
D:/中转站/scene_street_motion_converge_v1_motion/eval/step0006000/ema/
```

优先对照case004/011的同名普通视频，再看 `_GT_ASSISTED_ORACLE.mp4`。普通四列为RAW/AE/GENERATED/COPY；只有F4–F8是预测，COPY按设计静止。Oracle五列为RAW/AE/GENERATED/GT GEO+PRED TEX/PRED GEO+GT TEX。

审计与可复用脚本在 `D:/workspace/VggAE_DataAudit/scene_motion_20261022/`：

- `metrics_audit/analysis.json`、`analyze.py`：逐步和全量配对结果。
- `integrity_audit/integrity_metadata.json`、`video_decode_checks.json`、`README.md`：完整性、日志、SHA和解码。
- `oracle_audit/summary.json`、`findings.txt`、`encoded_video_difference.json`：分支/逐帧/异常例分析。
- `visual_review/final_atlas_00.jpg`至`07.jpg`：64例末帧对照。
- `visual_review/case004_roi_timeline.jpg`、`case011_roi_timeline.jpg`等：局部时序。
- `visual_review/case004_motion_oracle.jpg`等：带明确GT辅助标记的分支诊断。

本轮只审阅结果并记录结论；没有改训练代码或发起新训练。
