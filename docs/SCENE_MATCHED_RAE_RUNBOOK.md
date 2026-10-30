# Stage59：匹配特征源的视频 RAE 与固定窗口生成

2026-10-30。入口是 `scripts/scale/59_train_scene_matched_rae.sh`。
本轮回答两个具体问题：旧模型在固定训练窗口上能否自由生成；在共享视频 codec、
latent 形状和生成器配方下，DINO 前端与 StreamVGGT 聚合特征的结果是否不同。
这是新的受控小规模训练，不是 V-RAE 复现，也没有预先证明任一组会生成好画面。

## 启动与恢复

沿用原有 ModelArts 环境、三个节点各八张 910B、相同代码和环境变量，各节点启动：

```bash
bash scripts/scale/59_train_scene_matched_rae.sh
```

默认 `all` 顺序执行：

1. 24 rank 共享准备 RGB 与两类冻结特征。
2. 原模型 source31069 在固定训练池窗口、未见开发窗口上自由生成。
3. DINO 组：codec 两步保存 → 新进程恢复至 6000 步 → 冻结并缓存 latent →
   DiT 两步保存 → 新进程恢复至 10000 步。
4. Stream 组执行相同流程。两组顺序使用全部 24 卡。

无需另行提交 smoke；正式流程已含真实保存和进程重启。可选命令：

```bash
# 打印路径和阶段，不启动工作
SCENE_MATCHED_STAGE=dry bash scripts/scale/59_train_scene_matched_rae.sh

# 独立 _smoke namespace，24 train、4 dev，完整网络各训练 3 步
SCENE_MATCHED_STAGE=smoke bash scripts/scale/59_train_scene_matched_rae.sh

# 已完成 paired 数据准备后的训练/恢复，仍保留两步重启阶段
SCENE_MATCHED_STAGE=train bash scripts/scale/59_train_scene_matched_rae.sh

# 独立重跑历史模型 train/dev 自由生成
SCENE_MATCHED_STAGE=audit bash scripts/scale/59_train_scene_matched_rae.sh
```

同一代码、配置和数据身份中断后复用 namespace；会恢复 optimizer、每 rank RNG、
数据游标和 FP32 EMA。代码/配置/源权重改变时使用新的 `SCENE_MATCHED_NAMESPACE`。
数据 bank 一旦完成，已跳过名单也冻结；不会因重跑时某个视频恢复可读而静默扩充
训练集。要纳入修复的数据使用新 namespace，训练契约同时校验完整 bank manifest。
可用 `SCENE_MATCHED_ARMS=stream` 只恢复一组；这样单独运行不能代替双组比较。
`--stop-after 2` 是绝对更新停止点，不改变训练配方、学习率总步数或缓存身份。

离线资产继续使用已有 StreamVGGT checkpoint、LPIPS 包内标定权重，以及现有
`TORCH_HOME/hub/checkpoints/vgg16-397923af.pth`。不下载新 DINO 模型。若新节点没有
历史 VGG16 缓存，可用 `SCENE_MATCHED_VGG_URL` / `SCENE_MATCHED_VGG_MIRROR_URL`
指向已有 local/OBS 权重；不会静默去掉 LPIPS 或尝试公网下载。

## 数据与架构

| 项目 | 两组共同设置 | 差异 |
|---|---|---|
| 数据 | 复用 Stage58 的 512 窗口、64 开发窗口；相同坏数据跳过结果 | 无 |
| 固定训练评估 | 从整个训练子集按 seed 选 16 个不同视频 ID 的窗口 | 无 |
| RGB | 原 518 中心裁剪 RGB，bicubic/antialias 到 252，再量化 uint8 | 无 |
| 冻结特征 | 同一个已有 StreamVGGT checkpoint，patch-only | DINO 前端 1024 / Stream 最后层 frame+global 2048 |
| 可训练 bottleneck | 每 token LayerNorm、逐帧线性投影至 C256 | 输入/特征恢复投影宽度不同 |
| latent | 每帧 18×18×256，共 9 帧；无时间下采样 | 无 |
| RGB decoder | 共享卷积结构，两个时间注意力块，无独立 RGB encoder/texture 旁路 | 无 |
| 生成器 | 从头初始化 SceneFlow 831,305,344 参数，固定 4→5 帧 | 同初始化，不共享训练后权重 |

共同 decoder 为 13,580,403 参数。DINO codec 总参数 14,105,971，Stream codec
总参数 14,631,283；差异来自输入投影与特征读出。共同 decoder 在同 seed 下逐张量
同初始化，DiT 在 codec 构建/加载后重新设 seed，保持相同初始化与训练噪声顺序。
冻结 encoder 计算量并不相等，应另行报告，不能只用 downstream 参数宣称等算力。

这里的 DINO 是 **已有 StreamVGGT checkpoint 内的 DINOv2 前端**，不是独立官方
DINOv3 checkpoint，也没有照搬 V-RAE 的 EUPE/全部训练配方。本轮首先隔离当前
几何聚合前后的特征来源；双方都失败时不能据此断言“DINO 视频 RAE 也做不出来”。

每个窗口分别运行 full9 与 prefix4 冻结提取；缓存前四帧使用独立 prefix4 特征，
其余五帧为 full9 特征目标。保存两者前缀差异，不依靠宽松阈值宣称无泄漏。
逐帧 bottleneck 不混入未来。时间 decoder 可双向读取整个已观察+已生成窗口，
但生成条件只来自观察 latent；不会把完整 GT 未来交给 decoder 后再当作生成结果。

codec 损失为 RGB L1 + 0.5 LPIPS + 0.1 相邻帧差分 L1 + 0.1 特征重建 MSE。
选用固定第 6000 步 online codec，随后彻底冻结；不以验证 PSNR 选择更有利的一组。
归一化仅由各组 train latent 的 FP64 矩统计估计，dev 不参与。
DiT 为 x0 flow objective，logit-normal 时间采样，维度相关 time shift=4.5，
32 步 Euler 自由采样；未来初值为纯 Gaussian。第 0 帧作为 anchor，第 1–3 帧为
干净已知 token，第 4–8 帧被加噪和预测。不使用未来几何、caption、相机或 track。

AE 每卡 microbatch=1、accum=2，有效 batch=48；DiT accum=4、有效 batch=96。
BF16 AMP、梯度检查点、AdamW、grad clip=1、FP32 EMA；完整参数见
`configs/scene_matched_rae_v1.json`。6000/10000 是配对实验预算，**不是收敛判定**。
脚本记录每步实际时长和剩余训练估计，估计不含未来评估/上传；不要承诺固定完工时间。
无质量门槛阻止 AE→cache→DiT；缺失资产、损坏的已提交数据、非有限梯度等执行
错误会记录失败，保留阶段产物。单组失败不会取消另一组，最终退出码仍反映失败。

## 输出与判断

所有路径位于沿用的双输出根 `scale/` 下：

- `scene_matched_rae_v1/launcher/node{0,1,2}/`：阶段退出码、日志、有效配置、占卡日志。
- `scene_matched_rae_v1_data/`：paired 特征、bank、prefix 差异、原始窗口与跳过明细。
- `scene_matched_rae_v1_source_audit/`：旧模型 518 分辨率 train_eval/dev 自由生成。
- `scene_matched_rae_v1_{dino,stream}/codec/`：重建、latest/full resume、阶段 weights、selected codec。
- `scene_matched_rae_v1_{dino,stream}/cache/complete.json`：固定 latent bank 与 train-only 统计。
- `scene_matched_rae_v1_{dino,stream}/flow/`：生成器 checkpoint、weights 与自由生成评估。

新模型最终生成目录：

```text
scene_matched_rae_v1_dino/flow/eval/step0010000/{train_eval,dev}/{online,ema}/
scene_matched_rae_v1_stream/flow/eval/step0010000/{train_eval,dev}/{online,ema}/
```

每个 case 保存带标题的 `RAW / AE / GENERATED / COPY` 视频、generated.pt、metrics。
COPY 列恒定延续最后观察帧，这是明示基线。最终评估为 train16+dev64 × 两个 seed ×
online/EMA，共 320 个视频/组（有效数据减少时按 manifest 实际数量报告）。每 1000
步评估固定 train8+dev8、一个 seed、online/EMA；AE 同样保留阶段重建。
每 500 步保存 full latest 和 online/EMA weights；远端保留全部周期 weights，
本地只保留最近两份周期 weights，latest full 和 selected codec 不清理。

双读双写使用已有内容校验/receipt 链路，允许一端降级但显式报告。每节点共用一份
immutable feature cache，避免八个 rank 各拷整库。原视频下载受 12 GiB cache 控制；
本机测试/临时产物放 D 盘。占卡守护涵盖每节点八卡、CPU staging 和各阶段等待。
DI_throughput 分开标注编码输入 frames/s、重建/生成 tokens/s/NPU、采样
token×steps/s/NPU；包括对应读取和上传等待，占卡计算不进入训练吞吐。

看结果时按顺序判断：

1. 先分别看两组 AE 的可恢复画质。若新 codec 的真值重建就明显差，生成失败不能
   单独归因于 frozen encoder 或 DiT；下一步调整 codec，不能直接继续全 HQ 扩训。
2. 看纯噪声自由生成的固定训练窗口，再看 dev。训练集也失败与仅 dev 失败需要
   不同后续动作。历史 source 的 train_eval 是原训练池抽样，不是逐步访问日志证明。
3. DINO 成功而 Stream 失败，才有当前匹配配方下“几何聚合表示更难建模”的证据；
   两者均失败只能说明共同 codec/训练配方/预算仍不充分。两者均成功再讨论 VGGT
   几何独有收益与几何控制，不能只凭换 encoder 投稿。
4. 旧 518 模型仅作各自 train/dev 与重建差距参考；与新 252 的像素误差不能直接
   宣称提升百分比。新64个 dev 是重复使用的开发集，不是论文的新最终测试集。

RGB L1、运动差分、Laplacian 等是配对诊断代理，不等价 FVD、语义质量或几何正确性。
本轮不以单真值逐像素误差代替动态物体的可观看性判断。

## 本地验证范围

实际小 codec/SceneFlow 优化和恢复、真实双进程 Gloo 新进程恢复、未来真值负对照、
坏 RGB 共同跳过、双镜像恢复、正式入口参数及 Bash 阶段失败路径均有测试。
Python 编译、Bash 语法和 diff 检查在交付前执行。
本次 29 项联合测试通过；随后增加“完成 bank 的坏数据名单冻结”回归，数据和
训练器 11 项复测通过，合计 30 个不同测试。8 个 Python 文件编译和 Bash 语法通过。
本地测试使用小模型/合成输入和 CPU，不证明 910B 的算子、HCCL、真实 checkpoint、
OBS 吞吐、生产显存或画质通过；这些由正式流程的两步恢复和后续保存样本验证。
