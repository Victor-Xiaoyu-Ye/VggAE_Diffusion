# 为什么当前 R7 视频生成失败，以及如何获得第一个可信 baseline

日期：2026-09-10。范围：当前窗口式 I2V 实现、历次下载实验、用户提供的 26 个链接（去重后 24 篇）、MIRA 官方实现，以及补充的真实视频 scaling 工作。

## 结论与证据边界

**目前不能得出“VGGT/RAE latent 不适合视频 diffusion”或“从零训练必然不行”的结论。我们也没有证据支持继续按现有配方多跑一轮就能成功。** 已经完成的实验说明：当前约 178M DiT、约一万个固定窗口、单首帧条件、非线性压缩后的 R7 表示和现有优化预算，这个组合尚不能生成结构合理的视频。小幅换 shift、auxiliary 层位和采样步数没有改变这一事实。

最值得优先调查的是 **实际送给 DiT 的表示、预测任务难度和训练规模之间的匹配**。这不是把责任简单归给 AE：冻结 AE 的真值重建约 24.49 dB；它能重建，但生成模型产生的误差方向比等 RMS 随机误差更伤画面。新的本地分析还发现：通道标准化后仍有明显协方差集中，生成误差的空间频谱明显偏向高频。它们构成了具体线索，尚不是因果证明。

要获得 baseline，应先证明当前完整生成链条能在小集合上从纯噪声生成清楚、有运动的视频，再建立受控场景的严格 heldout I2V。小集合记忆只是工程验收，不是研究成果；但不做这一步，就无法判断该扩大数据、扩大模型，还是改变表示。原生预训练 I2V 可以提供质量参照，不能替代对 R7 路线的回答。

本轮没有启动集群训练、修改训练配方或声称 NPU 验证通过。新增的是可复现的 CPU latent 分析和研究结论。

## 1. MIRA 为什么是有效的反例，但不是我们的同配方复现

MIRA 证明了用表示自编码器和从零训练的生成模型获得 RGB 视频是可行的。它的主结果是 Rocket League 多玩家动作条件世界模型：10,000 小时、5B 模型，研究了规模变化和长时 rollout。这说明“没有预训练视频 DiT 就不可能成功”过于绝对；也说明只比较“都冻结了视觉 foundation encoder”会遗漏关键条件。[MIRA 论文](https://arxiv.org/abs/2607.05352)

本地官方仓库 `D:/workspace/mira` 对应提交 `db25448391d42547161673a77950a1154d8b5f1f`。以下具体实现来自该提交，不把发布默认配置等同于论文每一组实验。

| 项目 | MIRA 发布实现／论文 | 我们当前实际实现 | 含义 |
|---|---|---|---|
| 编码器和压缩 | 冻结 DINOv3-L，多层聚合后 strided Conv3d | StreamVGGT 多层特征、CompactCompressor、独立 texture 分支、投影、时空 codec | 都有压缩；区别是压缩结构和最终坐标，不是“它没压缩而我们压缩” |
| 单个 latent 时间步 | 配置推导为 9×16×32 | 18×18×192 | 每状态标量 4,608 对 62,208，相差 13.5 倍；不是 FLOPs 或任务难度倍率 |
| codec 目标 | L1、LPIPS、DINO consistency，自适应平衡 | 以历史 R7 重建成果为基线 | 需要检查压缩后的表示关系是否保留，不能只看冻结了源编码器 |
| decoder | 配置为宽 1152、深 28 的视频 ViT | 历史 R7 解码链条 | latent 难度与 decoder 承担的信息恢复任务要一起比较 |
| 预测条件 | 历史、动作；因果生成 | 独立编码的单首帧；一次生成四个 future latent slots | 我们必须同时决定未知运动和整段未来细节 |
| world model 训练 | 代码逐 latent 帧抽噪声，velocity flow matching；默认启用 shifted clean past | 全未来窗口共享噪声时间，weighted x0，整窗去噪 | 模型分解、条件与噪声协议不同，不能只抄一个 shift |
| 规模 | 论文主结果 5B；发布默认模型配置 1B | 当前约 178M | 规模可能重要，但没有证据说 178M 连受控短视频都必然不行 |

实现依据：[codec 配置](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/configs/model/raev2_codec_tdown.yaml)、[编码器](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/src/mira/codec/rae_encoder.py)、[world model 配置](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/configs/model/latent_world_model.yaml)、[训练和采样实现](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/src/mira/world_model/latent_world_model.py)。

我们应学的是：**把表示、条件、预测跨度和模型容量作为一个系统设计**。不能由此推出“压到 32 维就解决”“换成 diffusion forcing 就解决”。MIRA codec 配置的 `noise_tau` 是 0，不能把 decoder noise augmentation 描述成它成功的必要条件。它的逐步预测也不等价于本项目的单首帧整段预测；给我们的评估增加真实历史帧或未来动作，会改变任务。

补充检索找到更接近真实视频单首帧起步的研究：VATIX 从零训练 driving diffusion，区分模型容量、训练 exposure 和独立数据量。其结论支持按实测曲线决定预算，但它使用 Wan VAE 和远大于我们当前缓存的驾驶数据；不能把它“少于五个 epoch 内重复仍有效”的观察外推到我们约 58 次重复。[VATIX](https://arxiv.org/html/2608.28404v1)

## 2. 我们的完整链条：哪里已经验证，哪里没有

当前链条是：

`9 帧 RGB → StreamVGGT / texture encoder → learned compressor → stream projections → temporal codec → 标准化的四个未来 latent slots → 条件 DiT / flow → 反标准化 → frozen decoder → RGB`。

首帧条件单独编码，训练目标来自完整窗口。这一点必须保留。

### 2.1 真正生成的是 learned R7，不是原始 VGGT features

`models/dpt_latent_decoder.py::CompactCompressor` 对多层 VGGT 特征进行投影、卷积、归一化、非线性处理和空间池化；`models/causal_dual_tokenizer.py` 再投影 geometry / texture，并通过时空 codec 混合。源编码器冻结，并不意味着最终 192 维坐标保留了源特征的邻域、距离或几何可读性。

这不是已证实“压坏了”。MIRA 同样压缩，且能成功。真正缺少的证据是：**当前 R7 在取得好重建的同时，是否仍保有有利于生成的关系结构？** 应比较压缩前后局部对应、特征邻域与几何读出，而不是通过通道名字认定最后一段仍是纯 geometry。

SVG 的冻结语义分支加细节残差、VA-VAE 的表示对齐，提供了可检验的设计思想；它们不支持把任意重建 latent 统称为同等意义的 foundation representation。[SVG](https://arxiv.org/html/2510.15301v1)、[Reconstruction vs. Generation](https://arxiv.org/html/2501.01423v1)

### 2.2 AE 历史兼容错误已经修复，不能反复拿旧 bug 解释现在

历史 t2v2 checkpoint 使用跨时间 GroupNorm；曾经切换成 framewise runtime 后，虽然权重键能加载，重建下降到约 18.73 dB。恢复明确的 legacy contract 后，同一批 16 个视频的 AE replay 恢复到 24.48933 dB。后续 shift3 / aux 使用修复后的基线。

因此当前失败不能再归结为“没有正确加载 AE”。同时，legacy codec 的 future tail 归一化依赖整个窗口，**不具备可直接假定的前缀一致性**。同一前缀扩展不同未来时，其中间 latent 可能变化。对于当前整窗联合生成，这不等于未来条件泄漏；但它阻止我们直接把四个 latent slots 改成 MIRA 式在线逐步预测。

若做 autoregressive 版本，必须验证 `E(prefix)` 与 `E(prefix+suffix)` 对应状态是否一致，并明确 chunk 边界。不能只把旧模型的 norm 改成 framewise；之前已经实测这种静默修改会损害重建。

### 2.3 DiT 的表示宽度不是已确认的瓶颈

当前 `models/r7_window_dit.py` 为宽 768、12 层、约 178M 参数；每 token 输入 192 维，未来 4×324 个 tokens，全窗口 self-attention 和首帧 cross-attention，使用 clean-output head。768 已大于 192，**没有直接违反 RAE 论文讨论的 denoiser 宽度下界**。

RAE 的宽 head、位置处理、噪声尺度和训练时长可以作为控制变量，但不能从论文标题倒推“我们必然缺 DDT head”。aux6/aux8 只是对同一压缩 R7 target 的内部监督，不等于复现 REPA 的外部 teacher alignment，也不等于完整复现 RAEv2。[RAE](https://arxiv.org/html/2510.11690v1)、[REPA](https://arxiv.org/abs/2410.06940)、[RAEv2](https://arxiv.org/html/2605.18324v2)

### 2.4 flow 与 sampler 没有发现新的符号级错误

`utils/window_flow.py` 使用 `u=1` 为噪声、`u=0` 为数据：

`x_u = (1-u)y + uε`，`v = (x_u - predicted_y)/u`，积分从 1 到 0。

clean loss 除以 `max(u,0.05)^2`；除 floor 区间外，与该参数化下的 velocity error 对应。已有合成 oracle 检查支持时间方向和积分公式一致。真实采样从 64 增至 128 步只改善约 1.21% RAW L1，Heun 也没有带来结构性恢复，因此优先级低于表示和学习问题。

不能说“用了 x0 MSE 所以必然输出均值视频”：完整 flow 可以建模多峰分布。也不能在未训练 unconditional 首帧分支的模型上直接添加 zero-anchor CFG，视为正当修复。MIRA 使用 velocity，同样反驳了“velocity 天生不适合所有 RAE”的过度概括。

### 2.5 Wan 初始化快收敛，为什么仍然没有合理生成

`models/wan_compact_adapter.py` 明确绕过原生 `patch_embedding` 和 `head`，以新投影读写 R7；旧版本也有类似结构。这是在新坐标中迁移 transformer 参数，不能等同于继承完整的原生视频生成函数。

如果想证明“Wan prior 被保留”，应先有原生参考输出，再验证新接口在初始化时能否复现参考中间状态或预测场，并沿适配过程测量偏离。目前已有证据是收敛加速，不是这种函数保持。路线不是逻辑上不合理，但跨空间映射是主要研究问题，不能继续把加载成功当成迁移成功。

## 3. 新增实测：latent 尺度正常，但各方向并不等价

本次直接读取 stage34 下载的 `latents/*seed42.pt`，按 video ID 去重，使用 16 个视频各一个 seed。拼接后为 `[16,4,324,192]`，比较保存的 `target_normalized` 和 `generated_normalized`。未用这些 heldout 统计拟合或选择训练变换。

脚本：`audit_latent_structure.py`。机器可读结果：`latent_structure_audit.json`（与本文一起交付）。以下是 pooled descriptive statistics，不能当作全训练集的精确分布或实际内在维度。

| 通道协方差统计 | 结果 |
|---|---:|
| 最大主成分方差占比 | 17.16% |
| 前 8 个主成分方差占比 | 50.29% |
| 前 32 个主成分方差占比 | 68.94% |
| 前 64 个主成分方差占比 | 81.61% |
| 参与率有效秩 `(tr C)^2/tr(C^2)` | 17.32 |
| 熵有效秩 `exp(-Σp log p)` | 54.08 |

192 个通道分别标准化，不能去除它们之间的相关性。我们之前检查均值和标准差，覆盖的是边缘尺度，没有覆盖协方差结构。**这说明诊断不完整，不证明必须白化，更不证明 17 维就足以重建。** 小方差方向可能承载重要细节，全白化也可能放大噪声。

对 18×18 latent 网格做正交 FFT2，半径单位为 cycles/latent-cell；低频定义为 r≤0.15，高频为 r>0.3。阈值仅用于描述，未作为训练配方。

| 空间频谱能量占比 | target | generated | generated−target |
|---|---:|---:|---:|
| DC | 8.79% | 9.97% | 2.05% |
| 低频（含 DC） | 47.74% | 52.78% | 19.56% |
| 高频 | 25.34% | 20.52% | 50.21% |

误差能量的一半位于高频区，生成结果的相应能量又比 target 少。注意各列均除以自己的总能量；不能将占比差直接当作同量纲误差。latent 高频也不等于 RGB 纹理：必须经过 decoder 的频带替换试验才能知道它控制的是纹理、边界、几何还是别的内容。

这与 FreqWarm 值得联系，但不等于复现其结论。该工作强调 latent 频率与 RGB 频率不能混为一谈；其 warmup 是先对 RGB 做低通再编码，不能用“模糊当前缓存 latent”冒充。[FreqWarm](https://arxiv.org/html/2511.22249v1)

已有 decoder 扰动结果给出第二条独立线索：在完整的 12 视频×2 seed 配对中，匹配 per-slot latent RMS 后，alpha=1 的生成误差造成 RGB L1 约 0.0995，随机方向约 0.0527；alpha=0.1 时约 0.0122 对 0.00684。小扰动响应平滑，不支持泛化成“decoder 一碰就炸”；它表明**误差的方向有很大影响**。该下载的 metrics 仍是 306/384 行，不能声称整个作业矩阵完整；本次 16 个 latent 文件的统计与那 12 个完整扰动组是不同口径。

局部近似 `D(z+e)-D(z) ≈ J_D(z)e` 解释了为什么同 latent MSE 不代表同画面误差。下一步应测哪些子空间的误差影响解码，而不是立刻叠加一个新的 loss。

## 4. 数据适不适合：SpatialVID 的名字不是关键，实际训练暴露才是

当前 cache 入口默认每视频一个窗口，约 9,935 个训练窗口，每窗口 9 帧。6000 steps、global batch 96，总样本暴露 576,000，约等于 58 次经过这批窗口。它并不等于 576,000 个独立视频，也不等于充分使用了源视频全部时间段。这里不凭 9 帧反推总小时数，真实时间跨度应以采样 metadata 为准。

这些数据可用于受控短视频 baseline 和几何关系实验；目前没有证据证明足以从零得到广域、高质量的真实视频生成。单首帧无法确定相机将如何移动、人物做什么、遮挡后出现什么。相较于动作和历史条件，预测分布更宽；这属于任务本身，不是必然的数据错误。

应把数据扩充拆成两件事：增加同一视频的时间窗口，增加独立场景／运动覆盖。前者便宜但高度相关，不能当成等量新视频。所有 split 应按原视频或更强的场景去重隔离，再采时间窗口，避免相邻窗口跨训练和评估。

建议首先从训练来源构建一个明确的窄域（例如连续街景相机运动），筛掉镜头切换、重复、明显坏帧，记录 fps、时间跨度、相机与物体运动分布。不要只挑几乎静止的样本来制造好看结果；保留运动分桶和首帧复制基线。筛选用到离线未来统计可以是数据分析，但不能把未来信息送入测试条件。

## 5. 已完成实验告诉了我们什么

| 配方 | 最佳 EMA 步数 | RAW L1 | 判断 |
|---|---:|---:|---|
| shift3 无 aux | 2500 | 0.109971 | 保留主要对照，视频仍变形 |
| aux6，weight 0.5 | 3000 | 0.112138 | 没有改善 |
| aux8，weight 0.5 | 3500 | 0.109237 | 对最佳对照改善约 0.67%，不足以称质量突破 |

三组的后期表现均不能支持“再换一层就能成功”。但 6000 steps 也不是证明该模型已达到最优的理论预算。要区分容量、训练时间和数据不足，需要训练集合上的纯噪声自由生成能力，而不是仅看全数据训练 loss 或带噪真值重建。

单真实未来的 L1/PSNR 只能测 paired fidelity，不能完整评价随机 I2V 的合理性。首帧复制可能比真实但不同的运动得到更低 L1。我们观察到的结构崩坏仍是问题，但下一版验收要同时包含结构、运动、时序和分布评价，不能只按一列 L1 选“最好”。

## 6. 下一轮应如何获得第一个 baseline

### A. 先验证当前链条是否能学通；保留好的 AE

用同一个 `train_r7_window_diffusion.py`、同一个 legacy AE、同一个 9 帧目标、同一个首帧条件，建立固定 64 个训练视频窗口的记忆测试。训练时噪声和时间继续随机；测试每个窗口至少 4 个新 seed，从纯 Gaussian 开始，完整走正式 sampler 和 decoder。不得输入 GT future、不得从轻噪声 target 启动。

用全部 64 个样本做预览或可审阅汇总，防止只挑好例子。至少记录 RAW/EMA 两条生成曲线、分噪声区间去噪误差、latent 的频带／主成分误差、RGB 感知误差、每帧结果与运动幅度。既有 stage25/26 的 n1 历史试验和在大训练模型上看 train previews，均不替代这项同链条测试。

通过条件是完整自由生成保住轮廓、主体身份与连续运动，且与 clean AE replay 的差距显著缩小；不是只要求去噪 loss 降低。预算应预先设计算力上限，并按曲线决定是否值得延长，不能无限训练直到偶然出现一个好例子。

若 64 集合也无法接近重建，应进一步用单窗口定位，查优化、噪声分布、条件利用和实际梯度；此时优先讨论“数据不够”没有解释力。若能记住而 heldout 失败，表示/decoder 至少具有实现该小集合生成的能力，数据覆盖和泛化成为更高优先级。即使记忆通过，也不证明表示已经理想。

### B. 同时做 frozen oracle 干预；决定是否值得改表示

已有 target/generated latent 足够定义诊断，不需要训练新 AE。对生成误差 e，分别按空间频带和训练集 PCA 子空间做分解，构造 `z_repaired = z_generated - P e`，保持其余方向原样，再经相同 decoder。对照包括 target、generated、首帧复制，以及匹配被移除误差能量的其他方向；避免“修掉更多 MSE 当然更好”的混淆。

这是一项使用 GT 的离线 oracle，绝不能报告为可部署生成结果。它回答：修正哪一部分误差可以恢复主体／边界／运动？若某类小误差能显著恢复结构，训练目标和表示条件数值得重设；若需几乎完整替换 latent 才恢复，问题更接近整体分布未学会。

对源 VGGT 和 R7 再做对应关系／邻域保留检查；取训练统计定变换，以 heldout 验证。先测再决定是否采用保守预条件、分频预算或几何关系约束，避免一次引入多项无法归因的改变。

### C. 建立真正的 heldout 短视频生成基线

只有 A 学通以后，先以约 1–2k 独立来源、场景较一致的训练视频窗口建立窄域实验，保留严格 heldout；这个数量是用于诊断的起点，不是成功保证。仍然单首帧、9 帧输出，不给未来轨迹或真实历史作为额外条件。

按“记忆能力→窄域泛化→跨场景扩展”逐级增加任务复杂度。扩大数据时同时增加不同时间段和独立场景；根据现有模型的训练/验证曲线决定先增加 exposure 还是容量。不是把当前 178M 直接替换成 5B 并期待自动解决。

验收至少包含固定全量样本格、多个 seed、首帧复制对照、清晰度、身份/结构保持、运动量与时序一致性。小样本 FVD/FID 不能作为可靠的唯一结论；需要扩大 heldout 和独立来源数，并报告样本量。几何评价应有跨视角一致性/相对位姿等独立读出，且区分动态物体，避免只用同一 VGGT teacher 给自己打分。

### D. 若要借 MIRA 改造架构，应改变预测分解，而不是拼装关键词

候选方向是让 latent 具有可验证的前缀一致性，以已生成历史预测下一个短 chunk；显式保留几何关系，并让外观残差承担剩余细节。新的 bottleneck 应同时验收重建、源关系保留和小规模可生成性，不再只根据 PSNR 定案。

这是有条件的研究方向，尚不是本轮推荐立即大训的配方。旧 codec 不能直接在线复用；首帧独立初始化、chunk 对齐和训练/rollout 的上下文分布都要重新验证。动作控制可以另建任务，但不能悄悄改变现有 first-frame-only benchmark。

工程上继续保留双读双写、正式/缓存回退、完整 optimizer/scheduler/EMA/RNG resume、周期 checkpoint、训练与评估中间视频和 latent、DI throughput。新实验 namespace 与表示/噪声/数据 contract 绑定；不同表示或目标不做完整训练态混续。吞吐需要集群实测，不能由 MIRA 的 B200 数值换算成 910B 训练时间。

## 7. 这条路线的研究价值应怎样建立

“VGGT 替代 VAE”“加载视频模型并对齐几何”“加一个几何 loss”都不足以自动形成壁垒。更具体、可被推翻的假设是：**几何表示中对空间关系有用的信息，并不一定在重建驱动压缩后仍以适合随机生成的坐标存在；保留这些关系、减少生成对脆弱外观方向的依赖，能改善同算力下的视频结构和泛化。**

要支持它，需要等数据、等预算的几何 encoder 与语义 encoder 对照，以及压缩/对齐前后关系保留、decoder sensitivity 和 generation 的对应变化。DINO 系列是必要的有力对照，原生视频模型是质量参照。只有证明几何优势不是 AE PSNR、模型大小或额外条件造成，才有可信的 insight。目前这是假设，不是已经成立的新颖性声明。

## 8. 用户文献逐项判断

阅读范围：24 篇全部做了官方来源核查；核心相关工作读取方法正文，MIRA 另核官方代码；表中“摘要”表示本轮只用于任务定位与筛选，不声称完整复现或通读附录。重复的 VFMTok 和 VGGT-World 各合并一次。

| 工作 | 本轮深度 | 能借鉴的内容与适用边界 |
|---|---|---|
| [RAE](https://arxiv.org/abs/2510.11690) | 正文 | 表示维度、噪声与 denoiser 共同设计；当前 768>192，不能套用宽度不足结论。图像基准不能替代视频验收。 |
| [Latent Diffusion without VAE / SVG](https://arxiv.org/abs/2510.15301) | 正文 | 冻结语义与细节残差分工；重点是生成坐标如何保留表示，不只是 encoder 冻结。 |
| [4DLangVGGT](https://arxiv.org/abs/2512.05060) | 摘要 | 4D 语言与几何 grounding，可启发语义读出；不是 RGB 视频生成成功证据。 |
| [REPA](https://arxiv.org/abs/2410.06940) | 摘要及目标核查 | DiT 内部表示对外部视觉 teacher 对齐；与我们的同 R7 clean-head auxiliary 监督不同。 |
| [TexTok](https://arxiv.org/abs/2412.05796) | 摘要 | 语言承担部分语义，token 保留剩余视觉信息；需要合法语言条件，不能将 GT future caption 用于单首帧测试。 |
| [REPA-E](https://arxiv.org/abs/2504.10483) | 摘要 | 联合调整 tokenizer 和 diffusion；不是当前冻结 AE 上随加一个 loss 的直接证据。 |
| [Reconstruction vs. Generation](https://arxiv.org/abs/2501.01423) | 正文 | 优化重建可能损害生成友好结构，表示关系需要独立约束/验证；与本项目最相关之一。 |
| [Gen3R](https://arxiv.org/abs/2601.04090) | 方法核查 | 几何与外观协同，但保留视频 VAE 外观表示；不能当成纯几何 latent 替代 VAE 的同设置结果。 |
| [Repurposing GFM / GLD](https://arxiv.org/abs/2603.22275) | 正文 | 选择生成的特征层与后续冻结计算路径，值得学习；主要是有相机条件的静态多视图生成。 |
| [FreqWarm](https://arxiv.org/abs/2511.22249) | 正文 | 分析 latent 频率承载的信息与训练难度；RGB 低通再编码不等于模糊已有 latent。 |
| [REGLUE](https://arxiv.org/abs/2512.16636) | 摘要 | 全局/局部语义分工与融合；含 VAE 路线，不是直接证明当前 R7 足够。 |
| [Back to the Features / DINO world models](https://arxiv.org/abs/2507.19468) | 摘要 | 特征预测支撑感知和规划；不能以特征预测成功代替 RGB 生成质量。 |
| [VGGT-DP](https://arxiv.org/abs/2509.18778) | 摘要 | 几何特征作为机器人 diffusion policy 条件；生成的是动作，不是像素视频。 |
| [ViTs Need More Than Registers](https://arxiv.org/abs/2602.22394) | 摘要 | 提醒检查 patch/global 聚合及异常 token；没有证据证明我们的失败由 register 问题造成。 |
| [Beyond the Last Layer / DRoRAE](https://arxiv.org/abs/2605.10780) | 摘要 | 多层融合及能量约束有价值；本项目已经多层融合，增加层数本身不是答案。 |
| [Cubic Discrete Diffusion](https://arxiv.org/abs/2603.19232) | 摘要 | 高维 token 的离散生成分解；改量化与目标是另一条研究路线，不是当前连续 R7 的轻量修复。 |
| [Improved Baselines / RAEv2](https://arxiv.org/abs/2605.18324) | 正文 | 内部监督、模型/训练整体设置值得参考；aux6/8 不构成完整复现，细节应以代码核对。 |
| [SemanticGen](https://arxiv.org/abs/2512.20619) | 正文 | 语义规划再条件生成外观，将任务分解；最终还有 VAE 空间的视频生成阶段。 |
| [VFM Visual Tokenizers / VFMTok](https://arxiv.org/abs/2507.08441) | 摘要 | 语义保持与量化；自回归图像任务不同于连续视频 flow。 |
| [VGGT-World](https://arxiv.org/abs/2603.12655) | 方法核查 | 几何状态动态预测及噪声目标选择；不含可直接对标的 RGB 视频生成闭环。 |
| [FlowWM](https://arxiv.org/abs/2606.29059) | 正文 | 特征空间随机未来、时序辅助监督；主任务的感知预测质量不等价于 RGB 解码质量。 |
| [MIRA](https://arxiv.org/abs/2607.05352) | PDF＋官方代码 | 最直接的 RAE 视频成功参照；应比较瓶颈、条件、分解、容量与 exposure 的完整组合。 |
| [VideoRAE](https://arxiv.org/abs/2607.14088) | 方法核查 | 冻结视频基础模型、多尺度层级表示与压缩；说明 codec 需要为生成评价，而不是仅为重建评价。 |
| [V-RAE](https://arxiv.org/abs/2608.13556) | 正文 | 视频 latent、上下文条件和内部指导的设计；预测条件及数据集不同，遵守用户要求只借思想。 |

## 9. 不确定性与交付记录

尚未确定：协方差/频带结构是不是主要因果瓶颈；64 视频完整链条是否能拟合；当前模型扩大曝光或容量后的收益；压缩前后几何关系究竟保留多少；全量源视频时长和动态覆盖；新方案在 48×910B 的实际吞吐。这些都不能用论文成功替代本项目实测。

当前最强的结论是：**不要再把“好重建＋冻结 VGGT”当成已经验收的生成表示，也不要把当前失败当成 RAE 路线被证伪。先补齐可生成性和误差方向的因果验证，再花大预算。**

本轮新增：16 视频 latent 协方差/频谱 JSON 和可复现脚本；MIRA 配方对照；24 篇用户文献筛选；后续验证顺序。保留既有 AE、shift3 和 aux8 对照。没有声称能保证下一版生成合理视频，没有创建新训练入口，也没有改变正在运行的训练。
