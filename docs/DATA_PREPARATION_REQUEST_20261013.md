# 数据准备单：连续视频生成与几何验证分开

日期：2026-10-13。对应 [论文研究计划](PAPER_RESEARCH_PLAN_20261013.md)。

**当前不缺“再来几套RGB”，最缺现有RGB与时间、相机、来源分组的可靠对应。现有资产足够启动短视频基线，数据补充不应成为无限等待。** 本单只要求RGB帧/视频与必要元数据；训练目标仍是视频，不要求改成重建点云。

RGB继续放OBS。这里只检查目录、清单和小型annotation，必要临时文件放D盘；不把完整raw数据集下载到本机。

## 1. 现在实际使用的资产

以下是scene_rae_c256_15day_v1/production实际train cohort，非整个OBS库数量：

| 数据 | train来源 | 固定缓存窗口 | 原采样权重 | 新计划用途 |
|---|---:|---:|---:|---|
| DL3DV | 7,716 | 30,864 | 40% | 静态多视图/几何验证；有可靠时序才另作视频 |
| ScanNet++v2 | 962 | 3,848 | 15% | 表示与室内几何；DSLR照片不自动视为连续视频 |
| MVS-Synth | 96 | 384 | 2.5% | 暂存，不进入第一轮街景主训练 |
| OmniWorld game | 63 | 252 | 2.5% | 暂存，不扩大混训 |
| PointOdyssey | 131 | 524 | 5% | 动态结构/遮挡诊断 |
| Spring | 34 | 136 | 2.5% | 非刚性运动/细节诊断 |
| SpatialVID | 8,448 | 33,792 | 32.5% | 8,192街景+256replay；主线重新按父来源核验 |
| 合计 | 17,450 | 69,800 | 100% | 所选窗口缓存完整，**不是全HQ** |

这些“来源”未必都是独立物理场景，不能把数量直接当独立scene数。AE在线取样与flow固定四窗口的覆盖也不同。小域的高权重造成反复曝光，新增主线不会默认保留原七域比例。

主要现有路径：

- `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/SpatialVID-HQ/`
- `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/wai_dataset/`
- 旧重建库参考：`obs://yw-ads-training-gy1/data/external/x00445638/data/train_spatial/open_datasets`

## 2. 请优先准备的内容

| 优先级 | 数据 | 请补什么 | 不需要做什么 |
|---|---|---|---|
| **P0** | 当前SpatialVID-HQ街景及同域扩展 | 原CSV；`indexes.txt`、`poses.npy`、`intrinsics.npy`、`caption.json`所在路径与对应表；父视频/YouTube来源映射；缺失清单 | 不重新拉全量RGB，不为I2V手写caption |
| **P1** | 当前DL3DV | **与手头RGB处理版本匹配的**相机、原始frame-ID映射、resize/crop说明、scene split；优先旧6,118 train RGB-only部分 | 不假设另一套DL3DV相机可直接配当前帧；不重搬已有RGB |
| **P1** | ScanNet++v2 | 官方scene split；DSLR原图/去畸变版本、匹配相机与畸变参数；少量GT几何评估对应 | 不把照片顺序当fps；暂不要求所有点云 |
| **P2** | PointOdyssey | 保留原131/15/13 split；已有train131/val15足够开发，论文test若可取得再补13条及tracks/visibility/camera | 不重搬train；不要求先补全test才能启动 |
| **P2** | Spring | 原scene/左右camera/frame映射和可用GT；项目34/3/10与官方split对应 | 不索要官方未公开test GT，不把左右相机交错为时间序列 |
| 暂缓 | DynamicReplica、更多合成/重建库 | DynamicReplica原split找到后再议 | 用户已同意跳过，无需阻止其他训练 |

**现在最有价值的交付只有三项**：SpatialVID annotation+父来源映射；DL3DV相机与现有RGB版本的对应；ScanNet/Point/Spring原split和少量验证GT。其余按实验缺口补。

每个来源先提供20–50条有代表性的元数据记录和小annotation路径，确认适配后再导出全清单。无需先人工清洗全部数据，我们负责批量核验和坏样本记录。

## 3. SpatialVID主线怎么切

### 3.1 第一轮主域

选真实连续街景，优先步行/车载中一种主要采集形态，并在manifest标出另一种。先从现有8,192街景来源审计，256replay不混入主训练。白天、清晰、无转场作为初始质量条件，但保留实际行人/车辆等运动；避免为了“简单”只留下几乎静止的背景。

无人机自然景观留作独立域或后续扩展，不与街景在最初pilot中混合。若街景CSV无法可靠区分步行/车载，先做小样本抽查校准规则，不按caption关键词强行把所有样本标签化。

需要统计并分层：实际相机运动、动态物体比例、模糊/转场/压缩程度、真实Δt、有效pose覆盖、父来源重复。标签可来自metadata与自动测量，最后对代表性片段抽查，不能把自动筛选器的猜测当GT。

### 3.2 三档规模

| 阶段 | 建议规模 | 用途与边界 |
|---|---|---|
| 接口检查 | 每来源20–50条metadata，少量可读RGB片段由集群读取 | 检查格式/相机/时间；不是质量评估集 |
| pilot | 1,000–2,000 train父来源、8k–16k窗口；100–200 dev父来源 | 1/4历史→同4未来的短任务、表示对照 |
| 首轮主训练 | 利用当前8,192街景做深；按预算约100k不同窗口 | 避免每条长视频永远只固定四窗；不是承诺已存在100k合格窗 |
| 后续论文训练 | 可扩20k–50k同域父来源、100k–300k窗口 | 先证明新增来源可用且同域；当前数量尚未审计确认 |

同一长视频的不同窗口不等于新增独立场景。记录唯一parent/clip/window覆盖和曝光次数；可分批新增缓存，但各比较臂必须使用相同版本与刷新顺序。

### 3.3 最容易配错的标注帧

SpatialVID官方annotation约每 `int(fps/5)` 个原始帧取一帧，**pose数组第k项不是原视频第k帧**。`indexes.txt`是对应索引；先使用这些真实帧并记录时间，不直接把相机数组按RGB帧号切片。官方提供的内参是归一化形式，需转到对应RGB像素坐标，再应用相同crop/resize变换。[官方annotation说明](https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ#3-detailed-explanation-of-annotation-files)

先核验当前OBS处理版本是否保留同样约定。统一8Hz或任意轨迹插值属于另一个处理步骤，需验证后才启用。fps/时间未知时标未知，不编造秒数；pose缺失标缺失，不用单位阵冒充静止相机。

## 4. 其他数据如何使用

**DL3DV**：静态场景的多视图RGB与pose，适合排除物体动力学因素、做NVS与几何读出。即便原来源是视频，WAI导出的帧未保留fps时也不能当作已核验连续时间监督。当前8,051去重可用场景与7,716 train是不同口径。官方另有与训练场景分离的评估资产，若已持有可只补对应RGB+相机。[官方库](https://github.com/DL3DV-10K/Dataset)

**ScanNet++**：当前主要DSLR多视图用于室内重建/几何OOD。若后续需要第二个室内真实连续域，再准备少量iPhone RGB视频、timestamp和camera；匹配畸变/undistortion版本。[官方格式](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)

**PointOdyssey / Spring**：作为动态物体、非刚性、遮挡和跟踪的有GT诊断，初轮不负责补量。单camera连续流分开，同一父序列共享split。PointOdyssey官方v1.2为131train/15val/13test；Spring官方37train/10test，而项目当前34/3/10是在train中另留dev，不能混淆；Spring公开test不提供全部GT。[Point官方](https://github.com/y-zheng18/point_odyssey)、[Spring原文](https://arxiv.org/html/2303.01943v1)

几何评估先准备100–200个可验证pose/深度或轨迹的片段/场景即可，不要求训练集每个样本都有点云。原生几何head监督的伪标签与真实GT要在清单中分开。

## 5. split与论文评估资产

1. SpatialVID按原始父视频/拍摄来源切分。同一YouTube原片的不同shot、重编码副本、近重复片段、AE图像、caption和所有窗口必须同split。
2. DL3DV/ScanNet按scene；Point/Spring继承官方sequence split；左右camera/不同视角同父split。原held-out不因为扩容进入train。
3. 无parent-ID时，保留clip父ID并做近重复聚类，注明剩余泄漏风险；不是按窗口随机切完就认为独立。
   同时保存历史AE/flow训练清单及warm-start来源，逐阶段记录曝光。当前8,192街景已用于SceneAE，不能重新切出一部分就称未见test。新test优先取原held-out及未进入相关codec训练的新HQ来源；旧flow若见过该集合，其结果标为seen，不混作未见泛化。基础模型预训练中无法核实的覆盖另披露。
4. dev用于选配置；test用于固定最终报告。主域test建议300–500父来源，每来源2窗、2–3采样seed。分布指标另争取1,000–2,000个held-out片段；统计按来源而非重复窗口独立计数。
5. 再留一个真实域：可先从现有HQ审计出的独立采集类型做OOD，明确它仍来自同一数据集。要更强外部泛化，再选一套外部连续RGB；不要求同时准备四套。
6. 若日后主张回访/长期记忆，再专项准备50–100条含真实回访的连续片段。当前短I2V不需要先找这些数据。

## 6. 元数据交付建议

在OBS建立小型metadata前缀即可，RGB保留原URI：

```text
metadata_release_v1/
  README.md                 # 来源、版本、处理过程、坐标/尺度约定
  inventory.jsonl           # 每sequence或clip一条，不展开成巨型逐像素文件
  source_groups.jsonl       # parent/scene/近重复分组
  splits/train.txt
  splits/val.txt
  splits/test.txt
  missing.jsonl             # 已知缺文件/坏帧/标注未配对
  mappings/...              # 小型frame/time/camera映射，或引用现成annotation URI
```

示意记录（字段名可与现有清单适配；不是要求用户手工重写所有数据）：

```json
{
  "dataset": "SpatialVID-HQ",
  "version": "source-version-and-processing-version",
  "clip_id": "example-clip",
  "parent_id": "original-video-or-scene",
  "camera_id": "camera-0",
  "split": "train",
  "task": "temporal_video",
  "rgb_uri": "obs://bucket/existing/video.mp4",
  "rgb_size": [1920, 1080],
  "frame_map_uri": "obs://bucket/annotations/example/indexes.txt",
  "timestamp_uri": null,
  "fps": null,
  "time_status": "to_verify",
  "pose_uri": "obs://bucket/annotations/example/poses.npy",
  "intrinsics_uri": "obs://bucket/annotations/example/intrinsics.npy",
  "camera_convention": "to_verify",
  "translation_scale_source": "to_verify",
  "image_transform_uri": null,
  "caption_uri": "obs://bucket/annotations/example/caption.json",
  "has_pose": true,
  "has_caption": true,
  "annotation_alignment_verified": false
}
```

这里 `has_pose=true`只表示文件存在；在alignment核验前不能给训练valid mask=true。`to_verify`不会被当作已知坐标系，null不会替换为零运动。

I2V的最低字段是RGB、父来源、相机流、真实顺序、可核验时间/帧步长、split与连续段边界。pose/caption是对应条件分支的要求，不阻止RGB-only任务。相机分支额外需要K、pose、单位/尺度来源、坐标约定、图像变换和有效性。

## 7. 拿到数据后我们负责的验收

- metadata与对象路径抽查，计算实际可读覆盖；坏数据跳过并汇总，不让单个坏文件终止全轮。
- 多个视角/时间流是否混排、frame-ID偏移、pose是否对应同一RGB变换、重复帧/shot切换。
- 相机投影/相对变换的可视核验，归一化K与crop变换；先每源少量完整例再批处理。
- 父来源去重与split隔离；输出唯一来源、窗口、标注覆盖和缺失的机器可读统计。
- 选择少量真实低/中/高运动及动态物体样例，核对筛选标签；主训练不只留下容易例。
- 冻结版本/hash，列出实际被跳过的数量和原因；缓存完成比例与样本曝光同时回显。

这份检查用于获得真实可用清单，不设“可选标注不全就禁止所有训练”的门槛。严重错配样本不能进入相机实验，但可在RGB有效且split清楚时进入RGB-only任务。

## 8. 暂不要求的新数据

Cityscapes可用于短连续街景对齐，但官方sequence约30帧/1.8秒，不适合长期回访；KITTI Raw可补时间/标定明确的驾驶外部验证；RE10K只有在现有RGB可用时再匹配timestamps/poses，不能假设URL仍全部可下载；Waymo需选含RGB的对应版本，Motion轨迹库不能当视频库。[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/)、[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=road)、[RE10K](https://google.github.io/realestate10k/download.html)、[Waymo](https://community.waymo.com/open/about/)

优先补现有数据配套关系，再根据第一轮究竟缺第二真实域、长时间连续性还是控制GT，只新增一套能解决该缺口的数据。
