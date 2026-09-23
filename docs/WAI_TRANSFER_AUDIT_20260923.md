# 新传输 WAI 数据审计

2026-09-23。只读取 OBS 对象目录、大小、scene_meta.json 与少量处理日志；没有下载 RGB、视频、深度、权重或压缩包。审计产物位于 `D:/workspace/VggAE_DataAudit/wai_20260923/`。对象存在检查不等于图像解码或与原版逐字节一致。

## 新增动态数据的实测结果

根目录：`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/wai_dataset/`。

| 数据 | 已审计序列 | 非空 RGB 对象 | 元数据分辨率 W×H | 引用缺失 | 帧序列 |
|---|---:|---:|---|---:|---|
| PointOdyssey | 146 | 273,187 | 960×540 | 0 | 单序列 558–3,967 帧，编号连续 |
| DynamicReplica | 523 | 337,800 | 1280×720 | 0 | 503 组各 300 对、20 组各 900 对；左右分别连续 |
| Spring | 47 | 12,000 | 1920×1080 | 0 | 6,000 双目帧对；左右分别连续 |

716 个序列全部完成 JSON/LIST 检查。合计 622,987 RGB 对象，无小于 256 bytes 的可疑占位 RGB，无标记 `is_bad` 或 `_is_interpolated=true` 的帧。此结论仅描述当前元数据与对象，不证明上游从未删帧、重编号、重编码或插值。

### 确实处理过，但不是只剩点云

三个来源都是 WAI 布局：`images/` 保留 RGB，`scene_meta.json` 统一相机约定，另外存放派生标注。读取到的代表性 `_process_log.json`：

- PointOdyssey：conversion、covisibility、后续 cowtracker；增加 trajectory/flow 字段。代表性日志最后的 upload_obs 仍写 running，这是历史处理日志，不能据此判断本次传输仍在进行。实际全部 RGB 引用已对齐。
- DynamicReplica：conversion、covisibility。
- Spring：conversion、MoGe、covisibility；MoGe depth/mask 与原始 `image` 是不同字段。

日志记录的转换路径指向 MapAnything/WAI 处理脚本，但没有固定原脚本 commit，不能声称当前公开代码等于当时执行版本。公开 [DynamicReplica 转换器](https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/scripts/conversion/dynamicreplica.py)与 [Spring 转换器](https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/scripts/conversion/spring.py)中 RGB 采用软链接搬运、深度另行转换；这支持“格式转换为主”的解释，不能代替本批 RGB 的 checksum 对照。

### 是否删减或抽帧

**PointOdyssey：缺的是整个官方 test 分区，不是已传 train/val 大面积掉帧。** 当前目录为 train 131、val 15；对应 RGB 239,022 和 34,165。官方 v1.2 为 131 train / 15 val / 13 test。序列数与前两个分区一致，当前没有 13 个 test。所有本批序列编号从零开始、步长 1；没有发现保留每第 N 帧的编号证据。仍不能证明转换前后首尾时长或像素完全相同。[官方版本说明](https://github.com/y-zheng18/point_odyssey)

**Spring：场景数及总帧数与完整发布一致。** 47 序列中 37 个同时带 GT pose/depth，共 10,000 RGB；10 个没有 GT pose/depth，共 2,000 RGB。与官方 37 train / 10 test、5,000/1,000 stereo frames 对齐。测试 IDs 为 `0003,0019,0028,0029,0031,0034,0035,0040,0042,0046`；识别证据保存在 `spring_split_evidence.json`，并核对转换器只给 train 写入这两类 GT。[官方论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.pdf)

**DynamicReplica：完整性应按处理版本比较，不能简单判成漏了一个。** 原论文 README 写 524 videos、145,200 stereo frames；公开 WAI 数据卡写 523 scenes，恰好与本次目录数相同。本批实际有 168,900 stereo pairs，包含 20 条更长序列，不能把两个版本的宣传总数直接相减作为丢失数。当前全部引用均存在，未发现时间索引间隙，但原始 split 没有保留。用户确认暂时没有 split 清单，**本轮训练跳过它**。[原版说明](https://github.com/facebookresearch/dynamic_stereo)、[WAI 发布者说明](https://huggingface.co/datasets/ZhengGuangze/DynamicReplica_wai)

## 对训练的直接影响

1. DynamicReplica/Spring 的 frames 是 `left(t),right(t),left(t+1),right(t+1)`，不能顺序取 9 项当视频。加载器先选单相机、按数值索引排序，再选连续片段；双目两路继承同一个父序列 split。
2. PointOdyssey 只用 train 131 学习，val 15 留作评估；不伪造 test。Spring 用 34 个训练序列，`0013/0023/0037` 作项目验证，官方 test 10 保留。验证三序列沿用 MapAnything 的 Spring 列表，不宣称官方另有 val split。
3. 当前三批 scene JSON 没有 fps。新 WAI 视频取单相机连续编号，stride 1–2；不把它们宣称为一秒视频或用于物理速度标定。预览 8fps 只是输出播放速度。
4. 当前 RAE 仍只重建 RGB。depth、flow、trajectory、MoGe mask 不作为输入或 RGB 目标；相机仅为可选生成条件。`_is_interpolated=true` 将被动态加载器排除。

## 较早上传的静态数据：重要纠正

旧 `open_datasets/...processed_dl3dv_ours/` 的 6,378 组 RGB/相机配对本身完整，2,162,209 RGB。新 WAI 的 10,571 个 DL3DV 目录却只有 4,687 组具备至少 9 张 RGB：109 个 scene JSON 404，5,775 个已检查目录没有可用 RGB；另有 122 个部分缺帧的可用场景。缺失引用共 1,968,159，不能把场景目录数当作可用数据量。

新旧 DL3DV 去重可得 8,210 个候选父场景；重叠时优先完整旧 RGB，新增 1,832 个新 WAI 场景。首轮再排除其中短边低于 224 的 159 个极低分辨率场景，最终 8,051 个。旧相机与新 WAI 处理像素不能混配，所以旧版本当前只读 RGB。

WAI MVS-Synth 120 场景、12,000 RGB，无引用缺失；ScanNet++v2 1,006 场景、965,727 可用 RGB，无引用缺失，95,301 个 `is_bad` 标记帧已排除。ScanNet++ 使用内部场景级生成切分；官方 split ID 文件未取得，**不能用本次训练结果宣称官方 ScanNet++ benchmark 成绩**。官方 split 文件需授权下载；未绕过访问限制。[官方分区说明](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)

这些数据足够组织本轮静态/动态 RGB 表示实验。审计解决了可读取对象、分区和序列语义问题，不保证生成质量，也不替代集群上的 RGB 解码验证。
