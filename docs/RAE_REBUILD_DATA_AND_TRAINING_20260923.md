# VGGT-RAE 重建：数据已切分，先建立表示和生成的共同基线

2026-09-23。用户授权从已有重建数据与动态视频重新组织数据，借鉴 GAE，保持 VGGT 路线；确认暂无 RealEstate10K、ScanNet++ 或额外图文数据，先用现有资源。**本轮完成 OBS 元数据审计、候选切分和训练设计，没有启动训练。** 新数据不能直接传给旧 R7 trainer：它尚不支持这些多视图目录和新表示目标。

## 1. 已实际检查的内容

数据根：`obs://yw-ads-training-gy1/data/external/x00445638/data/train_spatial/open_datasets/`。

列出顶层 23 个目录，继续检查 DL3DV、MVS-Synth、OmniWorld、ScanNet、StaticScenes、ADT、Virtual KITTI 的内部布局；对主选数据执行完整帧文件名和配对检查。只读取目录、CSV、JSON 和一个 712-byte 相机 NPZ 样本，**没有下载 RGB、深度、视频、tar 包或权重**。审计与切分共约 32 MiB，保存在 D 盘。

| 数据 | 已列出的有效目录 | 检查结果 | 本轮用途 |
|---|---:|---|---|
| DL3DV 处理后 1K–7K | 6,378 | 2,162,209 个非空 RGB 文件与同名相机 NPZ 一一对应 | 静态/近静态多视图重建主源；静态生成对照 |
| MVS-Synth `GTAV_1080` | 120 | 每序列 100 帧；RGB、pose JSON、depth 文件名集合相等 | 小比例合成几何补充 |
| OmniWorld-Game | 196 UID | 920,458 个 RGB 文件；5,073 个 split 的索引均能找到 RGB 与 camera JSON；12,709 个 caption 文件 | 动态游戏候选，真实动态还需解码核验 |
| SpatialVID-HQ | 365,362 CSV 行 | 复用 2026-09-11 完整对象清单 360,610 个非空 MP4；新策略筛出 14,796 个候选 | 窄域真实视频和旧能力回放 |

检查边界：DL3DV/MVS 检查了每个相机文件的存在和非空，未读取每个文件的矩阵；OmniWorld 检查了全部 `split_info.json`，未证明每个 camera JSON 内的长度、坐标和数值正确。SpatialVID 对象清单不是 9 月 23 日的新 HEAD 检查。所有产物均标记 `training_ready=false`，不能把目录检查称为解码或 NPU 验证。

### 实际发现的路径陷阱

- DL3DV 的 `DL3DV/DL3DV-ALL-480P/` 是 HF 缓存形态，包括 snapshot、blob 和压缩副本；不作为当前直接加载根。
- RGB/相机在 `DL3DV_v1/DL3DV-ALL-480P-NEW/processed_dl3dv_ours/{batch}/{hash}/dense/{rgb,cam}/`。
- 深度等派生结果在另一个 `DL3DV_v1/processed_dl3dv_ours/{batch}/{hash}/dense/`。两套目录的 6,378 个场景 ID 对齐，是互补资源，**不是两倍训练数据**。存在 da3_depth 多版本，不自动选最高版本，不把它作为 VGGT 原生几何监督。
- 样本相机 NPZ 有 `intrinsic[3,3]` 和 `pose[4,4]`；不能由文件名推断 c2w/w2c、单位或当前图像尺寸。样本主点约 `(477,268)`，不可按“480P”名称硬编码图像/K。
- `MVS-Synth/Scene01...` 呈现 Virtual KITTI 的天气/视角布局，未纳入 MVS-Synth。只选 `GTAV_1080/0000...0119`。
- `scannet_v1` 的元数据是 ScanNet v2，不能当 ScanNet++；`staticscenes` 是另一个合成帧目录，也不能当 RealEstate10K。
- ADT 有动态活动，但原始 VRS/鱼眼转换和场景划分需要单独适配；其许多录制来自同一公寓/办公室，不能把录制数当独立环境数。当前不加入，避免同时扩大数据适配变量。[官方 ADT 说明](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
- OmniWorld 此处仅有 Game 子集；HOI4D 只发现大压缩包，没有为本轮解包。未把它计入可训练动态视频。

## 2. 冻结切分与数据角色

本地结果：`D:/workspace/VggAE_DataAudit/rebuild_data_20260923/selection_v1/`。

| 数据 | train | val | test | 分组单位 |
|---|---:|---:|---:|---|
| DL3DV | 6,118 | 128 | 132 | 场景 hash；132 包含 OBS 旧 test_index 中实际存在的 4 个 ID |
| MVS-Synth | 96 | 12 | 12 | 原始序列目录；这是项目自定义划分 |
| OmniWorld | 63 | 16 | 117 | 完整 UID；对应 657 / 158 / 4,258 个原始 split |
| 新 SpatialVID 街景 | 8,192 | 128 | 128 | clip ID；原始 YouTube source 未知 |
| 旧 SpatialVID 训练回放 | 256 | — | — | 仅旧 train_mixed 中未占用的 ID |

另保存 **512 个历史 eval/test ID** 的禁止训练表，用作持续回归评估；它们没有加入任何新训练集合。新街景 val/test 也排除了以前 single/mixed 小域训练用过的 ID。它们不保证未被更早的基础模型或 full-HQ 生成器见过，不能宣称所有旧模型都在完全未见域评估。

OmniWorld 官方 CSV 中 117 个本地 UID 带有 `Test Split Index`。我们将这些 UID **整个留作 test**，不把同一场景余下片段投入训练。由此剩余训练数据明显变少，这是当前资源的实际限制；不能用 5,073 个总 split 来宣传训练规模。跨 UID 的游戏/地图身份仍未知，也不宣称物理场景完全独立。[官方数据格式与元数据](https://github.com/yangzhou24/OmniWorld)

SpatialVID 策略固定在 `configs/rae_rebuild_data_v1.json`：白天、明亮、街景标签，短边 ≥480、时长 ≥4s、fps ≥8、美学分 ≥4.5、OCR ≤0.02、motion score 1–8、trajTurns ≤2。训练按 dynamicRatio 0.02–0.15 取 6,144 条、0.15–0.4 取 2,048 条。阈值用于构造可解释的初始队列，未用生成结果调阈值。

**dynamicRatio 只是标注代理；街景标签也不能证明第一人称步行。** 相机移动、独立物体运动、遮挡变化分别保存标签，不能相互替代。实际动态分层应在集群解码后用已有 mask/flow、补偿相机运动后的残差及抽查确认。极端运动/大量文字暂不进入首个生成域。

所有 T2I 帧、重建 views、视频窗口继承父级 split；禁止按帧重新随机划分。深度副本、处理版本、镜像地址都不能产生新的样本身份。官方 DL3DV 的独立评估集合与这里的 OBS 五项小 test_index 不同，本次是项目内验证，不宣称复现官方 benchmark。[DL3DV 官方说明](https://github.com/DL3DV-10K/Dataset)

候选清单还不是最终窗口清单：OmniWorld 的短 split 若不足所需帧数，记录跳过，不重复帧补足或跨 split 拼接。每个训练窗口保存原始 frame IDs 和实际时间间隔；同 UID 所有窗口共用父级归属。caption 必须按覆盖范围关联，不能把一个片段的文字套到另一个片段。

### 产物与重放

- `train_candidates.jsonl`、`val_candidates.jsonl`、`test_candidates.jsonl`：统一父级样本身份、真实 RGB/相机根、元数据状态、缺失条件和待检项。
- `historical_regression_ids.jsonl`：历史 512 个保留 ID。
- `excluded.jsonl`：主源排除表；本轮目录级检查未排除场景。SpatialVID 筛选数量另计于 report，不等于全 HQ 无坏数据。
- `selection.json`：最后写入的完整回执，包含策略、输入 SHA256、输出 SHA256、精确计数。输出目录非空时拒绝覆盖。
- `../probes/`：6,694 个审计回执。每项可恢复；成功回执是本次快照，不是永远有效的实时存在保证。

元数据检查脚本 `scripts/audit_rae_rebuild_metadata.py` 使用已有 SDK 凭据环境变量，不在代码/日志保存密钥。依赖 `esdk-obs-python`。它消费现有 listing 快照；不是一个自动下载完整数据集的工具。

```bash
python scripts/audit_rae_rebuild_metadata.py \
  --snapshots /path/to/rebuild_data_20260923 --dl3dv-per-batch 0

python scripts/prepare_rae_rebuild_data.py \
  --snapshots /path/to/rebuild_data_20260923 \
  --metadata /path/to/SpatialVID_HQ_metadata.csv \
  --inventory /path/to/obs_hq_video_inventory.json \
  --historical configs/spatialvid_domain_v2.json \
  --policy configs/rae_rebuild_data_v1.json \
  --output /path/to/fresh_selection
```

源快照命名是相对 `open_datasets` 路径以 `__` 替换 `/`；格式为 `items:[{prefix:...}|{key:...,size:...}],truncated:false`。主要输入为两套 DL3DV 的 1K–7K 清单、MVS-Synth/GTAV_1080 清单、OmniWorld RGB/annotation UID 清单、OBS `test_index.json`、[DL3DV-valid.csv](https://github.com/DL3DV-10K/Dataset/blob/main/cache/DL3DV-valid.csv) 和 [OmniWorld Game metadata](https://huggingface.co/datasets/InternRobotics/OmniWorld/blob/main/metadata/omniworld_game_metadata.csv)。官方可变 URL 的本次文件内容已在 selection 中固定 SHA256，更新内容须新建选择版本。

## 3. 新表示如何训练，才能解决之前没有解决的部分

以下是**拟实施的新模型合同，不是现有训练器已经具备的功能**。不再用修改旧脚本默认参数的方式改变历史 R7。

### 表示主候选

保留冻结 StreamVGGT 为主要特征源，重新训练可学习融合/压缩器、特征恢复器与 RGB decoder。当前 R7 训练冻结了早期 CompactCompressor 和 TextureEncoder；只在它们之后继续压缩，无法保证恢复已经损失的信息。新的训练边界需要向前移动到多层特征融合处。

候选从 `[4,11,17,23]` 四层出发，按训练集固定通道统计标准化，显式保留浅层细节与深层几何信息。**首版暂不做时域下采样**，每个 RGB 时刻对应一个 latent 网格；从单帧、短序列共同训练，让 T=1/T=9 都是原生输入。不要将 legacy R7 的跨时 GroupNorm 原地改成另一种 norm 再加载同名 checkpoint。

首个保真候选暂定 C256，保留 patch 网格；这只是待显存/重建验证的起点，不是“通道越多越好”的结论。C128 压缩是后续在同预算生成探针下比较的候选，不先按 GAE 的 C64 硬压。新 codec、归一化、分辨率或上下文改变，都使用全新的权重/缓存/统计身份。

RGB 必须仅由完整生成 latent 解码，不能把 GT RGB 或未生成的纹理侧路直接接入。历史 texture 分支保留为保真参考；若新版本仍需要显式外观分量，该分量必须进入生成目标，报告其贡献，不能把成功全部归于 VGGT。先测 VGGT 多层主候选的细节保留能力，再决定是否保留该分量。

### 监督不只看重建均值

1. RGB 像素与感知重建，分别报告域内 mean、分位数、每帧和 T=1/T=9；禁止一个域变好掩盖另一个域崩坏。
2. 恢复多层特征，检查由**同一 compact latent**读出几何的能力。当前仓库 `StreamVGGT` wrapper 只实例化 aggregator；有 DPT 源码不代表已经加载匹配的预训练 geometry head。先核验 checkpoint/head 键和原特征回放，再开启原头读出损失；不能挂一个随机 head 便宣称保留了 VGGT 几何。
3. 直接作用于生成坐标的空间关系监督，配合语义监督候选；不能只在一个额外 projection head 上对齐，就假设原 latent 已有好结构。teacher 类型、权重、坐标网格与特征标准化固定写入合同。GAE 的 C-RADIO/DINOv2 组合有消融支持，但不把其数值直接搬到 VGGT。
4. 静态区域可约束跨视图几何；真实动态区域按可见性/运动 mask 处理。禁止用静态 warp loss 强迫移动人车变成背景，也不把非连续多视图当等时间步视频强行平滑。
5. 在干净重建已过门槛后，单独加小幅 RGB-decoder latent 扰动鲁棒性训练，干净与扰动样本同时评估；几何/关系目标仍使用干净 latent。正则 warmup、开启步数、每项梯度规模都记日志；不一次堆满大权重监督。

训练初期建议按 **输入帧/token 配额**混合 DL3DV 45%、MVS 10%、OmniWorld 15%、窄域 SpatialVID 20%、旧 train 回放 10%。这是待实现的起始配比，非根据样本数量自然采样。OmniWorld 只有 63 个训练 UID，须限制单 UID 重复占比并记录每源 exposure/重用次数，避免少量游戏场景主导梯度。

AE 的单帧分支训练 RGB 编解码能力，不依赖文字；静态序列训练多视图恢复；动态序列保留真实帧顺序、时间间隔及运动细节。三者不要用“静态视频”一个标签混为一谈。先 1/4/9 帧短序列，长序列延后；StreamVGGT 每场景重置状态，参考条件始终独立由可见帧编码。

## 4. “先 T2I 再 T2V”应如何执行

**先训好同一 RAE；生成先做短程单图学习验证，随后尽早进入单图+视频共训。** 不主张先投入巨额预算做一个通用 T2I 模型，再突然换任务；也不从第一天就只学视频。

[GAE 论文](https://arxiv.org/html/2609.24981v1)区分 codec 与 flow 两阶段，flow 使用 T2I 共训；它不是给我们“必须完整单图预训后才能训练视频”的证据。公开 codec 配置中单图共训开关又与 flow 不同，不能混为一谈。

我们当前的文字资源也不同：DL3DV/MVS 路径中未确认 caption，OmniWorld 文字按片段覆盖且 2 个 UID 未发现 text 文件，SpatialVID 要关联原 annotation。**缺 caption 的图像仍可训 RAE，不能冒充 T2I 样本。** 不用文件名或宽泛场景类别代替描述。仅用视频抽帧可建立任务域内单图基线，不能称为获得大规模图文预训练先验。

建议顺序：

| 阶段 | 要回答的问题 | 进入下一阶段的证据 |
|---|---|---|
| 数据适配与 AE 基线回放 | RGB、K、pose、帧索引是否同一个样本 | 集群解码/坐标检查；旧 R7 同 RAW/518 参考回放；固定 train/val/test |
| 新 RAE | 干净 RGB 是否保真，geometry 与动态细节是否仍可恢复 | 分域不退化、T=1/T=9 均正常；latent 统计/扰动审计 |
| 冻结 RAE 的单图生成探针 | 静态图像本身是否能从噪声生成 | 同批 latent 上 train fit 与 heldout 图片均可辨识；固定 seeds/CFG |
| 短视频共训 | 增加时间轴后是哪一环退化 | 9 帧 + 持续单图任务；DL3DV 静态/窄域街景动态分别报告 |
| 扩展 | 提高分辨率、时长和域数能否保住质量 | 上阶段通过才逐项扩展，不同时更换 AE、DiT、数据和采样 |

没有文字的单图 unconditional 小集实验可用于检查扩散能否拟合，但必须叫做无条件拟合测试，不能当 T2I 成功。T2I 正式 warmup 以已经核验的图文配对为前提。后续视频阶段保持约每 3 个视频任务插入 1 个单图任务作为起点，分别记录实际 token/样本曝光；同一模型空间模块共享，不能训练两个无关模型再把提升归于共训。

几何生成先使用可控的相机轨迹，将相机运动与物体运动区分，并保留 null-camera 对照。参考图像 clean memory 与未来 noisy targets 分离，时间约定、CFG/dropout 和训练采样保持一致。不从待生成画面反推条件后再送回模型。WASD 可以将来积分成轨迹，但这批数据没有动作干预标注，当前只能验证相机控制，不能承诺通用可交互物理世界模型。

## 5. 不掉点与显存、工程合同

- 旧 `r7_t2_c192_v2` checkpoint 和 legacy runtime 永久保留。相同 RAW、裁剪、分辨率、时间索引下逐样本比较；旧域 24.49 与新街景 22.41 的历史差异已经证明不能跨 cohort 套一个 25dB 阈值。
- 在新 DL3DV/MVS/OmniWorld 队列先测旧 AE，获得该队列真实基线；目标仍是提升至可靠保真水平，不能通过换数据让平均 PSNR 看起来更高。
- 候选晋升同时检查 RGB PSNR、LPIPS、结构/纹理缺陷、时序与几何。建议初始回归容忍范围：paired PSNR 不低于参考 0.1dB、LPIPS 不劣于参考 2%，同时检查尾部样本；阈值为工程起点，先估计重复评估误差，不自动放宽以通过。测试集不参与调参。
- 保存 `best_reconstruction`、`best_joint`、periodic、latest、EMA/optimizer/scaler/scheduler、rank RNG、sampler cursor；只有评估通过的 checkpoint 才生成新的冻结合同。
- 252 分辨率约 18×18 patch，9 帧共 2,916 tokens；518 约 37×37，9 帧共 12,321 tokens。后一档 tokens 约 4.2 倍，全局 attention 成本可能约 17.9 倍。**不照搬 GAE 的 81 帧高分辨率配置到当前 910B。** 先单卡实测最坏批次，再考虑 3×8；52 GiB allocated gate 沿用工程上限起点，不能以参数内存代替实测峰值。
- AE 要在目标高分辨率上训练/评估；生成的低分辨率探针是单独的 compute gate。不能把低分辨率图像放大后与旧 518 重建直接比 PSNR，也不能偷偷改变 encoder 输入网格后复用 latent stats。
- 数据适配统一记录 timestamp、原始 RGB 尺寸、crop/resize 和变换后 K、pose convention、scale 来源。DL3DV COLMAP/处理后 pose、MVS extrinsic、OmniWorld metric factor、SpatialVID 估计轨迹分别校验，禁止一律乘同一尺度。RGB 几何映射一致，缺少可信 pose 的样本只能进入不需要该条件的任务，不把单位阵当真实位姿。
- OBS 输出仍为平台路径与 owner 路径双写，优先可用副本双读；模型合同一致才能 resume。原始数据只有一个已核实来源时不伪造第二来源。中间 manifests、坏数据 ledger、评估和预览先落盘再提交；坏训练窗口可显式跳过但不静默补抽，固定评估失效须记缺失并阻止可比性结论。
- 后续所有 NPU 编码/训练按原约定输出 `DI_throughput: <value> tokens/s/npu`，排除通道数、分清数据时间和计算时间，记录 rank、active NPU、任务与实际 token 数。**本轮只有 CPU 元数据检查，报告 records/s，不虚构 DI token throughput。**

## 6. 研究方向与当前边界

近期主张是：建立既能保住几何和动态外观、又能被扩散实际学会的 VGGT 表示。是否适合生成，需要固定预算的 RGB 生成来检验，不能只靠重建 PSNR 或 latent MSE 宣布成功。后续差异性可以来自因果观测条件、动态区域与静态几何的分工、生成误差下的几何可读性；这些尚不是已证明的新贡献。

本轮新增了可复现切分策略、完整目录级审计与候选清单；没有新增或启动 AE/DiT 训练器，没有改写 stage49/50，没有刷新它们的实验结果。下一项实际实施是新数据 loader 的集群解码/相机合同验证与新 RAE 主候选，随后才交付可启动训练脚本。避免把一份配置文档包装成已经能训练的系统。

验证记录：新增切分/元数据检查测试及相关旧数据工具测试共 7 项通过；全量真实元数据重新生成一次，selection 与全部文件 SHA256 完全一致；Python 编译、`git diff --check` 通过。`selection_v1_replay/` 为约 20 MiB 验证副本；清理操作被工具自动审批拦截（仅返回 blocked by policy），暂留 D 盘，不是另一套切分。未进行 RGB 解码、几何读出或 Ascend 训练验证。
