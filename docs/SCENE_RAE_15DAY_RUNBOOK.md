# 三节点 Scene RAE：表示、图像、视频的完整训练

2026-09-23。用户要求只用重建数据中的 RGB，静态与动态混训；3×8 Ascend 910B、约 15 天完成各链路，阶段保存和双读双写；质量门槛不能阻止后续阶段。本文件覆盖早先“设计尚未实现”和“PSNR/T2I 硬门槛”的状态。代码与 CPU 验证完成，**尚未在 NPU 启动，未保证画面质量**。

## 实际数据与切分

固定 manifest 为 `configs/scene_cohort_v1.json`（约 6.4 MB，仅路径、分区和元数据）。详见 [完整数据审计](WAI_TRANSFER_AUDIT_20260923.md)。不会把 scene 数量当作互不相关的物理世界数，也未核实不同 YouTube ID 的原视频归属。

审计工具 `scripts/audit_wai_rgb.py` 读取分源目录快照，凭据只从环境取得；新增来源使用 `--datasets pointodyssey dynamicreplica spring --report dynamic_rgb_summary.json`，`--refresh` 可重查。Spring 检查会生成分区证据。`scripts/prepare_scene_cohort.py --previous <旧selection_v1> --wai <本次审计目录> --output <manifest>` 合并、去重并保留旧split；输入SHA写入manifest。正式启动直接使用仓库内已冻结manifest，不要求集群先重做整个目录审计。

| 来源 | train | val | test | 训练抽样比例 |
|---|---:|---:|---:|---:|
| DL3DV 新旧 RGB 去重 | 7,716 | 165 | 170 | 40% |
| ScanNet++v2 DSLR | 962 | 25 | 19 | 15% |
| MVS-Synth | 96 | 12 | 12 | 2.5% |
| OmniWorld Game | 63 UID | 16 UID | 117 UID | 2.5% |
| PointOdyssey | 131 | 15 | 无 | 5% |
| Spring | 34 | 3 | 10 | 2.5% |
| SpatialVID 窄域街景+旧 train 回放 | 8,448 | 128 | 128 | 32.5% |

DynamicReplica 按用户最新回复跳过，原因是原始 split 未恢复；不阻止本表其余数据训练。PointOdyssey/Spring 只取单相机连续 RGB，保留官方 held-out；ScanNet++ 是内部生成切分，未验证官方 benchmark split。旧 SpatialVID 512 个 eval/test ID 均不进入训练。DL3DV 旧版本相机处理未核实时相机条件为空，不拿新 WAI 的 K 配旧像素。

空间输入统一中心裁剪到 518×518，WAI 的 K 使用同一裁剪。读取像素尺寸与相机元数据不符、无 pose 或姿态非法时取消该样本的相机条件，继续 RGB 学习。静态多视角只监督重建，不把照片编号当固定视频时间；真正视频才计算 RGB 时间差分。WAI 动态序列无 fps，stride 1–2 不赋予“一秒”含义；SpatialVID/Omni 依据 fps 采约一秒。

AE 按源权重在线读取不同窗口；冻结缓存按父场景 4 个窗口、全局 hash 混排、24 ranks 分片生成。36 小时到期时提交已成功部分并继续，manifest 写 `complete_source_pass`。初始上限约 70K 训练窗口，实际以 manifest 为准；这不是全 36 万 SpatialVID 视频重跑。

## 表示与生成架构

`RGB → frozen StreamVGGT 多层特征 + 可训练纹理编码器 → 可训练压缩/投影 → 每视图 [18,18,256] → RGB decoder`。

- C256=geo128+texture128，不做时间下采样，9 视图保留 9 组 latent。纹理也是生成目标，decoder 没有 GT RGB skip。
- 从 `r7_t2_c192_v2/joint/checkpoint_best.pt` 加载已知空间 compressor、texture encoder、RGB decoder 与投影；投影扩大通道时新增 decoder 列初始为零。**不加载 R7 时间 codec，也不把旧 legacy GN 强改成 framewise 再冒充原 checkpoint。** 旧完整 legacy R7 作为同帧重建参照单独回放。
- StreamVGGT 冻结。compressor/texture LR 1e-5，其余 trainable LR 5e-5；AdamW、warmup500、clip1、BF16 autocast、activation checkpoint。
- 目标为 RGB L1 + .5 LPIPS + .1 四层归一化 VGGT 特征恢复 + ramp 至 .1 的直接 latent 空间关系 Gram + .1 动态 RGB 时间差分。特征恢复目标是 pooled18×18 多层特征；**不是已验证的原生37×37几何头恢复**。
- 首帧由独立单帧 encoder 调用得到，训练/缓存均替换 joint 首槽，保留源条件契约。T1 每4步出现一次；RGB decoder 保留源模型允许的时间 attention，但没有时间压缩。
- 65% AE 时间预算后逐步加入小幅 latent jitter，最大 .05×每通道 std，隔次用于 RGB 分支；干净 latent 保持特征/关系监督。
- 选择 domain-balanced `best_joint.pt`（RGB L1+.1 feature+.1 Gram）冻结；同时保存纯重建最优、最终和阶段快照。新 codec 重训仍可能掉 PSNR；每域 paired old-R7 delta 会明确报告，不能保证继承旧25dB。

生成器为共享图像/视频的 DiT：768宽×24层主干，1536宽×6层去噪头，12 heads，**831,305,344 参数**。复用本仓库经过测试的 WindowBlock/SDPA 和 WindowFlow；从零训练，未加载 Wan DiT 权重。首帧条件走独立 memory，未来 8 帧是噪声状态，相机只在已验证 WAI 数据上给每token ray；无条件是显式缺失。

Flow：train-only 每通道标准化；`z_u=(1-u)z+uε`，预测 x0，loss `||(x0-z)/max(u,.05)||²`。logit-normal 时间按单视图维度计算 shift=`sqrt(18²×256/4096)=4.5`，不乘帧数。采样从 u=1 到0，32-step Euler，FP32积分、BF16网络，固定 seeds，CFG=1 作为首轮受控输出。

图像阶段没有参考图像条件；有现成真实 caption 的 SpatialVID 使用已有 UMT5 embedding，其余做无条件单图学习，**不能把无 caption 的照片称作 T2I 数据集**。视频阶段每4个更新有1次单图任务，其他做首帧条件8帧预测；text/ref/camera dropout=.1/.2/.2。ray 采用首帧相对坐标、单位平移基线，最后两维为平移存在/相机存在，不把 metric 与 COLMAP 任意尺度混成绝对距离。

## 向 GAE 学习的边界

核对官方代码 commit `a61ebe542ae777bb6c87d8262e311122f00894e7`，仅阅读，没有复制其 academic-only 实现或下载权重。[GAE 源码](https://github.com/TencentARC/GAE-GeometricAutoEncoder)

借鉴：多层特征恢复、直接作用 latent 的关系目标、延迟且只作用 RGB 的 noise augmentation、宽去噪头、图像/视频共享训练、clean reference 与被积分变量分离。官方 codec 配置的 `cotrain_t2i` 关闭，而 flow 开启；不要把两阶段混为一谈。其图文外观数据、DA3/frozen geometry head、C-RADIO/DINO教师和长序列预算都与我们不同，本版不声称 GAE 复现。

还发现 release 中 shift 注释与值不完全一致：dim82944/base4096 的平方根是4.5，配置 pin 为6.364；shift>1 实际把本项目定义的 u 推向更高噪声。明确数学契约，未盲抄注释或 pin 值。18网格是对当前预训练空间模块与3节点预算的取舍；早先37网格建议未直接照搬。

## 15 天预算与显存

| 阶段 | 上限 | 内容 |
|---|---:|---|
| AE | 72h | 静态/动态图像重建、表示约束、旧模型对照 |
| cache | 36h | 24 ranks 缓存 latent、训练统计和固定验证样本 |
| image | 24h | 共享 DiT 单图学习 |
| video | 180h | 继承 image 权重，图像/视频混训 |
| 余量 | 48h | 初始化、完整短跑、保存尾部、重启恢复 |

训练阶段总预算312h=13天，另留2天；wall cap 在同步 optimizer/shard 边界检查，包含阶段内评估/IO。初始化与最后评估保存可跨越单阶段 cap，不能当作严格360h硬截止或保证步数。AE/image/video另设100K/100K/200K更新上限，先到者结束。重启继承已花时间与消费 cursor，耗时不会重置。

AE microbatch1，accum2，全局48 clips；Flow视频 microbatch1，accum4，全局96 clips；单图 microbatch8，accum4，全局768 images。视频噪声2592tokens，单图每microbatch也是8×324tokens，但注意力成本不同。831M参数用FP32权重/梯度/Adam状态+BF16 EMA，基础状态约14GiB/卡，另有DDP桶、激活和rank0解码器等；这**不是峰值估计的实测值**。启用activation checkpoint和NPU SDPA，仍要真实短跑记录 `peak_allocated_gib`。

## 启动与恢复

继续使用已有 ModelArts 三节点镜像、环境、代码/StreamVGGT 依赖准备命令；三个节点执行同一入口。平台注入 `VC_WORKER_NUM=3`、`VC_TASK_INDEX`、`VC_WORKER_HOSTS`，或使用既有显式分布式变量。不要把所有节点的 NODE_RANK 都设为0。

```bash
SCENE_NAMESPACE=scene_rae_c256_15day_v1 \
SCENE_STAGE=all \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/51_train_scene_rae_15day.sh
```

入口与 cwd 无关。前置资产：历史最佳 R7、StreamVGGT、可加载的 LPIPS/VGG 感知权重与既有 Python 依赖。默认从固定 owner OBS 读 fullhq UMT5 bank；WAI 不虚构 caption。文本 bank 按当前 cohort 在节点共享目录缓存小子集，同一原始 shard 不由8个rank反复远端下载，原始SHA先验证，大源shard释放；第一次访问仍可能较慢。明确要完全关闭文本才设 `SCENE_TEXT_ROOT=none`，不可无声降级。

`all` 自动依次完成 rehearsal/ae→cache→image→video，再执行 production 全四阶段。rehearsal 使用同一完整网络、RGB形状、24 ranks、正式梯度累积，trainer各2个更新、cache每rank2样本，保存并真实恢复optimizer/RNG，再采样。质量差不拦截。技术错误如维度不匹配、OOM、全数据不可读会直接暴露，避免72h后才发现下阶段代码不能执行。

若只想运行这个短跑，使用 `SCENE_STAGE=rehearsal`；后续 `all` 会读取完成标记继续。正常中断重启直接重复原命令，已完成阶段自动跳过。单阶段重试可选 `ae/cache/image/video`。同namespace要求数据/config/world不变；补数据或改模型要新namespace，防止旧latent冒充新表示。

质量不达标不触发停训；坏RGB有限重试并写失败ledger跳过，同步各rank取可计算batch。连续128次全局无可用batch、非有限loss/grad、两个持久化目的地都失败属于无法继续计算/保全状态的错误，不是PSNR门槛。DataLoader硬故障、进程被kill或集群掉电仍可能终止作业；重启恢复最后已提交checkpoint，未提交尾部会重做。

## 双写双读与输出

owner镜像：`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/scene_rae_c256_15day_v1/`，同时写平台OUTPUT_URL的对应目录。节点临时目录位于 `/cache/yexiaoyu/vggae_runtime/outputs/scale/scene_rae_c256_15day_v1/`。RGB按需缓存每节点200GiB，可用`SCENE_RGB_CACHE_GB`调整，不整库下载。

- `launcher/node*/pipeline.log, exit.json`：启动器stdout持续双写。
- `production/contract.json`：config、cohort hash、输入身份、world、代码hash（代码hash为记录，不作为阻止兼容修复的门槛）。
- `production/{ae,image,video}/checkpoint_latest.pt`：每小时、评估后、结束保存，包含optimizer/cursor/RNG/spent；`checkpoint_final.pt`保存阶段终态。
- `production/ae/best_reconstruction.pt, best_joint.pt, complete.json`：最优重建/联合选择及实际冻结版本。
- `production/{ae,image,video}/weights_step*.pt`：周期快照；远端保留，成功双写后本地只留最近8份。
- `production/cache/state-r*.pt`：每rank缓存cursor、moment统计、shard清单；`cache/complete.json`为冻结训练索引，`eval.pt`保存固定RAW+latent。
- `production/{ae,image,video}/eval/step*.json`及`samples/step*.mp4`：每3小时与最终评估。AE列为RAW/AE，生成列为RAW/AE/GENERATED。每域4例、同一固定种子，不拿pixel L1当自由生成成功的唯一标准。
- `metrics-r*.jsonl`、`tb/`、`publication_status-r*.json`、`{ae,cache}/bad-r*.jsonl`：数值指标、TensorBoard、双写退化记录、坏数据记录。MP4使用macro_block_size=1，保持518，不暗中放大528；视频编码失败会写error.json，优化器继续。

模型/shard先写payload再写SHA256提交receipt。读取校验字节，从另一路恢复；最新checkpoint恢复先由leader统一提交版本，避免不同节点读到不同optimizer状态。至少一路持久化成功才算提交，单路失败明确记录，两路皆失败报错。SIGTERM尝试在安全边界保存；不能保证平台极短kill宽限内写完数GB，定期保存因此必要。

所有训练/cache打印 **`DI_throughput: <value> tokens/s/npu`**，JSON有同名数值。计数不含channels、不乘world；AE按输出grid，cache包含独立anchor编码，flow只数noise target tokens；各自分母是该段walltime（含IO/等待/评估），不能混成纯算子吞吐比较。

## 已验证与仍待验证

11项CPU测试：小型实际codec/flow跑通RGB→AE→cache→image→video；optimizer/RNG恢复；双进程混合任务/累积/共享目录发布；坏主副本备用读取；新receipt覆盖旧本地optimizer；caption子集复用；直接latent关系梯度；空reference不读anchor内容；相机裁剪/尺度不变性；stereo与时间拆分；父场景/旧512ID保护等。老师特征、LPIPS和视频渲染在集成测试中使用fixture，不冒充真实模型推理。

Python编译、shell语法和diff检查属于静态验证。真实910B峰值显存、HCCL、预训练权重、MoXing写入、实际RGB解码，以及重建/生成质量，由集群rehearsal及后续正式训练验证。本版目标是交付可恢复且可观察的完整实验，不是宣布唯一根因已确定或15天一定得到高质量世界模型。
