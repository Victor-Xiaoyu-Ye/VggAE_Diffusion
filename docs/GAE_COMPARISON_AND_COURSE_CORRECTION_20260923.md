# GAE 对照与项目推进纠偏

2026-09-23。用户提供 GAE（arXiv:2609.24981），追问为何该工作成功而本项目尚无合格生成基线。已读正文、附录、官方配置与本项目实际训练代码。本次没有训练、下载权重或刷新集群结果。

## 应当承担的判断问题

当前结果不足以归因为用户目标错误、VGGT 路线不成立、DiT 太小或 SpatialVID 天然不能用。Full HQ 的生成器已是 1,652,968,512 参数。参数量不是完整的模型能力或训练预算指标，但继续用“需要大模型”作默认解释已经缺乏依据。

此前修复 AE 兼容性、采样公式与缓存问题有必要；问题在于随后多轮实验仍主要围绕固定 R7 调整生成器，而没有尽早完成同条件下的成功表示对照。9 月 10 日的文档已指出压缩后结构保留、协方差集中和生成误差方向问题，但这些线索没有及时转为针对表示训练目标的生成实验。知道可能的问题，不等于解决了它。

上一轮转向场景运输、证据修正的研究故事，是进一步的能力假设，不能代替最初的 RGB 生成基线。GAE 是应优先复核的直接正对照；此前调研漏检了这篇直接相关工作。现在应先获得有证据支撑的基础结果，再决定差异性方向。

## 论文的关键证据与范围

[GAE 正文与附录](https://arxiv.org/html/2609.24981v1)以 DA3 为基础。受控对照约 0.93B，9 个 252×252 视图，8 GPU，训练计划 100K 步；最终 81 视图展示另用 80 GPU、多域数据训练。两者不能混为同一预算。

表 8 的 FVD 消融：表示监督 266.4→233.4；参考条件处理 257.1→225.7；位姿尺度处理 350.7→225.7；T2I 共训 472.9→225.7。它们是各自实验块内的结果，不能跨块相加、当作对本项目的因果分解或承诺相同收益。

这些证据同时反对两个简单解释：只要扩大 DiT 就够；只要增加一个 latent 正则就够。论文支持的是表示、条件和外观学习配合的完整方案。它也没有证明我们换成某两个 loss 就一定成功。

## 代码层面的具体差异

官方源码只读 checkout：`D:/workspace/VggAE_DataAudit/GAE-GeometricAutoEncoder`，commit `a61ebe542ae777bb6c87d8262e311122f00894e7`；只取源码和配置，未拉取权重和媒体资源。公开[代码](https://github.com/TencentARC/GAE-GeometricAutoEncoder)及[权重入口](https://huggingface.co/TencentARC/GAE-D64-1B)可作为下一步复核基础，本地尚未运行推理。

| 核对项 | 本项目实际路径 | 官方方案可借鉴之处 |
|---|---|---|
| AE 优化 | `train_causal_dual_tokenizer.py:707` 包含 RGB、geo/tex 特征、时序差分和 latent 正则；不能写成仅 RGB 训练 | [codec 配置](https://github.com/TencentARC/GAE-GeometricAutoEncoder/blob/a61ebe542ae777bb6c87d8262e311122f00894e7/configs/gae_64.yaml)同时配置冻结几何读出、token 对齐、直接作用于 posterior mean 的关系监督 |
| latent 正则 | `train_causal_dual_tokenizer.py:176` 约束每通道均值/标准差 | 边缘尺度约束没有保证空间关系和适合生成的表示组织 |
| 版本边界 | stage49 固定 `r7_t2_c192_v2`；当前训练源码中的 geo-motion cosine 是后续版本加入 | 不可把今天源码已有的 loss 倒算为历史 checkpoint 训练过的目标 |
| 表示来源 | 多层 StreamVGGT 经学习压缩及 geo/tex 时序混合，固定 legacy codec | 原特征来源并不证明最终生成坐标保有原模型几何读出的能力 |
| 生成器 | stage49 width1536/depth24；x0、uniform、无 aux head | [flow 配置](https://github.com/TencentARC/GAE-GeometricAutoEncoder/blob/a61ebe542ae777bb6c87d8262e311122f00894e7/configs/flow_gae64.yaml)使用 DDT 分层宽度与 depth；参数总量不能替代架构适配对照 |
| 外观训练 | Full HQ 使用视频窗口与 caption，尚无单图生成共训 | 官方 flow 配置明确启用 `cotrain_t2i`；加载文本编码器不等于获得生成先验 |
| 相机 | stage49 不提供相机，stage50 才是有界 pose/null 对照 | 官方配置有逐 token 射线条件；当前 pilot 的 pose MLP 不等价于此 |

与论文的参考条件消融比较时，不能无证据说我们的 source 条件读取未来、或套用同一种 ODE clamping bug。本项目已有独立首帧编码和固定条件路径。需要精确区分“参考特征的上下文契约”“表示时序混合”“相机条件”，不能把它们统一称为泄漏。

具体核实：stage49:86 开启 `--independent_anchor`；`cache_causal_video_latents.py:531` 仅编码 `frames[:, :1]` 构造条件；`train_r7_window_diffusion.py:156` 强制检查缓存标记。`models/r7_window_dit.py:161` 的噪声输入只含 future，anchor 是额外 memory；`utils/window_flow.py:50` 也仅积分 future。不能把它写成论文消融中的参考槽 ODE clamping。当前 trainer 的未来预测循环未实现 T2I 共训分支。

官方 release 也需要按实际配置复核：其 `docs/METHOD.md` 的 pose normalization 注释与当前 `flow_gae64.yaml` 中 `pose_translation_norm: none` 存在表述差异；公开配置是长序列 release 配方，不能直接称为论文所有受控消融的完整复现配置。应固定 commit 和 checkpoint 元数据后核查实际执行路径，而非只复制 README。

## 纠偏后的下一步建议

1. 以官方完整 checkpoint 做固定少量样例的推理正对照，先确认我们自己的运行和评估链条能复现其基本能力。DA3-GAE 成功不算 VGGT 实验成功，但比另一轮无参照的大训更能缩小问题。
2. 在这个成功参照下建立受控的几何 RAE 训练基线；明确哪些来自公开权重、哪些从零训练。优先保留完整生成配方及源条件契约，避免只挑一个正则、shift 或相机 MLP。
3. 若继续保持 VGGT 为主表示，改变 encoder/codec 时重新训练适配的 bottleneck/decoder 并重建缓存、统计和契约；保持生成任务、条件和评估一致。先证明 RGB 基线，再做运输或在线修正的创新。

以上是建议顺序，不是新训练已批准、已实现或已通过 NPU 的报告。旧 full-HQ 与 stage50 不在本轮被停止或改写。工程上继续保留 OBS 双读双写、中间结果、断点和字面 `DI_throughput: <value> tokens/s/npu`。

## 不应再作的承诺

- 不因某篇方法成功，就断言本项目唯一根因已定位。
- 不因当前失败，否定几何基础模型表示生成的方向。
- 不以 latent MSE、好 AE PSNR 或几何读出替代合格 RGB 生成。
- 不把重建指标、条件信息量、数据更新量不同的实验直接排名。
- 不在基础结果尚未成立时，以更大的 world-model 叙事代替交付。
