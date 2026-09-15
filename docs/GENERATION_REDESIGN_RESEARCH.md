# VGGT 视频生成的失败分析与项目重设计

当前证据支持停止继续微调现有 R7-window 配方，但不支持判定 VGGT 表示不能用于视频生成。更准确的判断是：项目还没有建立一个在当前数据、任务和算力下有效的生成系统，因此不能把失败唯一归因于编码器、数据量或采样器。

下一阶段应保留“只给首帧，生成合理未来视频”的主任务。先建立完整预训练 I2V 的质量参照，再用一个受控的表示对照决定是否重建 R7 生成空间。相机轨迹可以成为以后独立的控制能力，也可以用于解释任务难度；它不能替代首帧生成的成功标准。深度、分割等任务头适合检验表示内容，不能替代 RGB 生成评估。

研究方向建议收敛到一个可被推翻的问题：**在相同的首帧信息和训练预算下，几何表示能否减少需要重新生成的信息，同时保留外观质量？** 这比“把 VGGT 用作 AE 编码器”更有意义，但目前仍是假设，尚不是已经建立的论文贡献。

本报告的本地证据截止于 stage47 完成；外部资料覆盖至 2026-09-15。方法建议尚未实现，未启动新训练。论文结果均为作者报告，不代表已在 Ascend 环境复现。

## 1. 实验事实与结论边界

### 1.1 已经确定的事实

最新完整单域训练使用 2048 个视频、每个视频 4 个固定窗口，共 8192 个窗口；模型约 178M，6000 步，全局 batch 96。累计 576000 次窗口呈现，相当于每个缓存窗口平均约 70.3 次。这个数字衡量重复训练量，不能当成独立视频数量。训练、缓存、重放和最终轨迹的证据见项目实验报告。[^1][^2][^3]

| 证据 | 结果 | 能支持的判断 |
|---|---|---|
| 同样本 AE 历史兼容性重放 | 历史控制约 24.489 dB；新街景约 22.412 dB；缓存/重新编码误差在很小范围内 | 已修复的 framewise/legacy 不兼容不是本轮持续失败的主要解释 |
| 最新单域 EMA6000 | RAW L1 0.126551；首帧复制 0.126585 | 没有建立明显优于简单参照的像素预测能力；不能只靠此指标评判所有合理未来 |
| 最终 32 视频 × 2 seed 轨迹 | one-call RAW L1 0.121820；step8 为 0.119202；最终 0.126551 | 采样过程中存在误差累积，但已测中间点的标量改善有限 |
| 靠近真值的去噪探针 | u=0.015625 时，latent MSE 0.000313，RGB L1 对 AE 0.002267 | 真值附近局部去噪有效；输入已有 98.4375% 真值，不能视为生成成功 |
| 已查看的 one-call、中间和最终画面 | 纹理重复、建筑变形、树木重影、涂抹 | 没发现一个可用的隐藏高质量结果，仅改终止步数缺乏支持 |
| mixed 分支 | 尚未真正训练 | 不能声称单域与混域对照已证明数据分布的因果作用 |

轨迹共 1792 行、48 个完成状态、6 个成功退出和同步回执。可视检查覆盖 3 个视频的若干帧及一个 seed，不能外推为逐一检查全部输出。step8/u=0.875 虽有最佳均值，却不在保存的对照网格中，不能声称已经直接检查它的视觉质量。[^2]

历史 memory64 实验说明模型至少可以学习部分训练样本结构；它没有形成良好的 held-out 生成。它也不支持“模型完全不工作”或“换采样器即可解决泛化”这两种结论。AE 重放的严格一致，关闭的是特定兼容性问题，不是对整个数学设计的认证。

### 1.2 仍然不能确定的原因

目前缺少“相同数据、相同生成任务、相近生成器预算，仅更换表示”的成功控制。因此，容量不足、独立训练样本不足、R7 目标分布难学、条件接口不充分、decoder 对生成误差敏感，这些解释仍有混杂。

“单域也失败”只能说明当前单域配置没有挽救系统。街景标签缩窄了语义类别，却没有固定相机高度、镜头、前进速度、建筑纹理、遮挡、转向及曝光；它并没有变成同一个游戏场景的反复交互。与此同时，训练还从原先约 1 万视频缩到了 2048 个视频，不能把这次尝试理解为增加了数据支撑。

更值得承担责任的设计问题是：此前过早把 AE 重建指标当作固定表示的充分依据，然后把大量迭代集中在 diffusion loss 与采样时间上。今后应把“真实特征可解码”“生成特征可解码”“条件分布可学习”作为三个独立验收项。

## 2. 当前模型实际在学习什么

### 2.1 R7 已经不是原始 VGGT 表示

当前流程包含 StreamVGGT 多层特征、空间压缩、geometry/texture 两个投影分支，以及学习得到的时序 codec。多层特征来自 4、11、17、23 等层；空间网格由 37×37 压到 18×18；两路各投影到 96 通道，再经过 192 通道的时序混合。对应实现包括 `models/causal_dual_tokenizer.py`、`models/dpt_latent_decoder.py` 和 `models/causal_temporal_codec.py`。

首帧单独编码为条件，后续 8 帧被压成 4 个时序 latent，每个为 18×18×192。一次生成 1296 个 token、248832 个标量。这个数字说明目标的具体规模，但不能直接换算为相对其他模型的计算难度。

geometry/texture 的分支名称不保证压缩后的通道仍然具有可独立解释的几何/纹理含义。投影、归一化和跨通道时序操作可能重新混合信息。若没有压缩后的几何读出或关系保持证据，就不能宣称 DiT 正在一个已经被验证的几何状态空间中建模。

类似地，StreamVGGT 特征会依赖输入历史，legacy 时序 GroupNorm 又引入整窗统计。这个固定窗口 AE 可以用于联合窗口生成，但它不满足任意前缀扩展时 latent 不变的要求。直接切片、增加 memory 或切换归一化，都不等价于获得正确的自回归 tokenizer。

### 2.2 像素预测误差和视频生成质量没有一一对应关系

首帧无法决定真实视频下一秒的唯一镜头轨迹、行人行为和遮挡内容。合理的生成可能与记录下来的未来不一致，首帧复制反而可能获得较低 L1。因此 RAW L1 应保留为配对预测诊断，而不应继续单独决定“最佳生成 checkpoint”。

当前 flow 定义为 `x_u=(1-u)y+u*epsilon`，由 x0 预测构造速度 `(x_u-pred_x0)/u`。在未触及 floor 的范围，x0 MSE 除以 u² 与相应速度 MSE 等价。这种训练可以学习多模态分布；不能因为单次 x0 的平方损失最优解有均值性质，就把整个多步 flow 等同于确定性平均预测。

但 u=1 时的单次预测处在极低信息输入上，更平滑的结果可能降低配对误差。最终细节增加而 L1 变差，既可能包含不正确结构，也可能包含与某个真值不同的合理变化。当前可视证据确实显示明显变形，因此这里不是纯粹的指标误判；只是指标使我们更难正确选择改进方向。

### 2.3 加载 Wan 权重为什么没有保住质量

旧适配把 R7 接到 Wan 的内部骨干，同时替换或绕开原生 patch embedding、输出 head 等接口。迁移后内部张量的含义、噪声分布、条件方式及解码器都变了。损失更快收敛说明预训练参数有用，但不说明原生视频分布被完整迁移。

真正能回答这一点的参照应保留原生 VAE、输入输出投影、图像条件、文本接口和采样过程。官方 Wan2.1 提供 I2V-14B；T2V-1.3B 不能自动当成原生 I2V-1.3B。官方 CUDA 示例也不是 Ascend 已验证实现。[^4]

因此，“再加载一次 Wan”不是新方案。应先测完整模型，再对接口变化逐步建立可归因的质量差异。如果需要适配，关闭适配分支必须重现原生输出；只检查权重加载成功不够。

## 3. 相关工作改变了哪些判断

### 3.1 MIRA 的成功条件

MIRA 报告使用 1 万小时 Rocket League 游戏数据、5B 生成模型，并结合历史状态及多玩家动作。其价值包括动作归因和物理交互评估，不能只概括为“RAE 加多玩家故事”。论文的模型/数据规模实验说明规模有实际影响，但这些实验没有隔离我们与它之间的编码器差异。[^5]

公开 codec 配置使用冻结 DINOv3-L、多层聚合、32 通道 bottleneck、时空各 2 倍压缩，以及宽度 1152、28 层的 decoder；训练结合 L1、LPIPS 和 DINO 特征一致性。在 288×512 输入下，一个压缩时序位置为 9×16×32。[^6]

由此得到的设计启示是：不能把“MIRA 也是 foundation encoder”理解成“我们这个 R7 加一个 DiT 应有同样难度”。它对表示、解码容量、状态条件和训练量做了成套设计。当前模型却同时承担从首帧猜相机运动、保持静态结构、补充新区域和生成复杂纹理等任务。

也不能反过来简单断言游戏容易。高速物理和多主体作用本身很难；有限场景重复与可观察动作降低了某些不确定性，同时引入了其他难度。比较应该围绕具体条件分布，而不是“真实数据一定难、游戏一定简单”。

### 3.2 VideoRAE 与 V-RAE 不是小预算单帧预测的直接证据

| 工作 | 与当前决策相关的原文事实 | 对本项目的限制 |
|---|---|---|
| VideoRAE | 受控 T2V 为 2B、800K steps、global batch 256；高质量展示另来自 11B OpenSora2 在 5M 视频文本样本上的适配 | 不能把展示样例归因于一个小型从零 DiT；两种设置须分开[^7] |
| V-RAE | Cityscapes 使用第 4–15 帧预测第 16–27 帧，配合专门的视频 decoder；研究重建与时间表示/生成的关系 | 12 帧历史与当前单首帧不同；可借鉴评估与 tokenizer 设计，不能搬用结论[^8] |
| RAE | 图像 foundation 表示配合 decoder、针对高维表示的生成器设计 | 支持可行性，不证明任意学习压缩空间都继承其效果[^9] |
| RAEv2 | 区分全局语义与局部空间结构，结合中间表示监督；其 patch 输出仍使用 LayerNorm | 归一化本身不是通用失败解释；当前 aux6/8 也不等价于完整方法[^10] |

VideoRAE 受控实验的样本呈现量是 204800000 次，当前为 576000 次，约相差 356 倍。这是 batch×steps 的算术比较，**不是独立数据量比、训练 FLOPs 比，也不是本项目所需最小预算**。它足以否定“类似训练量下别人已成功”的暗含前提，却不足以给出扩大到多少就会成功的承诺。

### 3.3 后层归一化、细节丢失和生成空间几何

应把三件事分开：任务训练造成的表征偏好；LayerNorm 等数值操作；经过当前 learned compressor 后的实际目标分布。任务头能读出深度，不等于纹理细节完整；低重建误差也不等于生成模型容易采样出这种表示。

多层融合工作 DRoRAE 报告改善重建，但同时强调融合造成的分布变化会影响生成兼容性。SVG 则保留语义分支并增加细节残差，其消融显示未做分布对齐的细节增强反而损害生成。两者共同提醒我们：增加信息与降低生成难度并非同一目标。[^11][^12]

一篇更直接相关的新工作在 VGGT 四层、任务头归一化之前的标准化坐标上建立球面流匹配，使用目标相机条件、约 50M 模型，在 59103 个场景训练。其欧式/球面对照有改进，但仅生成单个目标视图，还对 RGB head 做了生成特征适配。[^13]

**对本项目的推断：** 这使“几何表示的数值结构”成为具体可检验假设，却不能推出 R7 应立即改球面 flow。LayerNorm 在忽略 epsilon、去除 learned affine 后才近似对应固定范数的零均值约束；后续线性压缩和非线性时序 codec 通常不会原样保持这个集合。必须在实际 target 上测 token 均值/范数、方向与幅度的解码贡献，再决定模型结构。

此外，满足范数约束只是满足一个必要结构，并不意味着落在真实数据分布上。把生成 token 投影到球面，可能修复幅度，也可能保留完全错误的方向。训练分布也不能只在推理时临时更换。

RAEv2 使用归一化特征与欧式生成仍取得好结果；上述球面论文的实证优势应理解为特定 VGGT 接口及任务上的结果，不能升级为“所有归一化特征的欧式 flow 必然失败”。这也是不直接照搬其摘要中强结论的原因。

### 3.4 几何 foundation model 的直接生成已有多种形式

GLD 以 DA3 为主、VGGT 为扩展，针对多视图和目标相机生成多层特征；它选择生成层的边界，并用 cascade 保证跨层兼容。它还明确区分仅从源图编码的条件和经过联合注意力的完整目标特征。[^14]

Gen3R 联合建模外观和几何，但 RGB 使用预训练 VAE 解码，几何分支衔接 VGGT。VGGT-World 预测未来几何特征并交给原有几何头，目标不在于高质量未来 RGB。FlowWM 主要通过下游感知任务评估未来特征。[^15][^16][^17]

这些工作分别证明“几何特征可以生成”“外观与几何可以结合”“未来几何可以预测”，没有共同证明“当前 R7 能在 2048 视频、6000 步下生成好视频”。不能把不同任务的成功混成同一种 baseline。

### 3.5 需要明确排除的弱创新

| 候选表述 | 已有直接覆盖 | 当前判断 |
|---|---|---|
| 用 VGGT 特征监督视频模型 | Geometry Forcing；CamGeo | 不能单独作为贡献[^18][^19] |
| 生成语义/几何，再生成外观 | SemanticGen；Gen3R | 需要新增机制和实证，而不是模块替换[^15][^20] |
| 几何检索历史，减少重绘 | VMem；CovRAG；Lyra 2.0 | 已有 visibility、coverage 与 correspondence 设计[^21][^22][^23] |
| 错误几何降权、无有效关系时退回原生生成 | TourPhysics | 简单 confidence gate/残差回退不足以立项[^24] |
| 在 latent 内进行几何奖励 | VGGRPO | 几何奖励或 latent geometry head 本身已被探索[^25] |
| 尊重 latent 流形并增强 decoder 稳健性 | Riemannian VGGT；GAE | 需要针对当前表示提出不同且可验证的问题[^13][^26] |

Lyra 2.0 尤其接近“几何只负责关系、视频模型负责外观”的建议：其方法已经使用几何对应、覆盖检索和生成历史增强。因此这句话适合作为工程分工，不能再包装为独有 insight。[^23]

## 4. 数据是否适合

SpatialVID 的真实街景和自然视频适合做短时 I2V、场景保持、相机运动相关的几何一致性研究。其空间标注使它比只有视频文本的数据更方便，但标注来自估计流程，不能自动视为标定真值。[^27]

完整 HQ 元数据约 36.5 万行、此前核对的有效视频对象约 36.1 万；当前训练并未使用这一规模。新单域分支 2048 个视频的完整片长合计约 6.5 小时，4 个一秒窗口的名义采样量约 2.28 小时，且存在相关性。完整 HQ 的存在为扩展提供可能，不代表这个子集足以从零训练高保真开放世界生成器。

当前 CSV 分域适合作为可复现抽样，但存在三条边界。首先，Street Scene 不保证第一人称步行，不能仅靠 sceneType 得到视点类别。其次，clip ID 不重复并不保证源 YouTube 视频、地点或连续片段不泄漏。最后，运动字段的单位和意义未全部验证，不能把分位数阈值解释成真实速度与角度。

建议继续使用 SpatialVID，暂不换数据集来追求表面成功。评估沿用冻结 cohort 以保持可比；扩展训练从完整 HQ 进行，把清晰度、切镜、动态比例、前进/转弯、视点等做成显式分层。CSV 用于本机筛选和清单核验，真实视频与几何标注的检查、去重和缓存都在远端完成，不把 raw 数据下载到本机。

单域的作用是建立更容易理解的学习曲线，并非把视频筛得越相似越好。需要同时保留不同地点、不同纹理、不同运动的 held-out；最终应报告按源视频/地点隔离的结果。在缺少源映射时明确报告此限制，不能宣称跨场景泛化已经建立。

## 5. 项目重设计

### 5.1 先区分三个交付物

**可看的生成参照**回答“同样首帧上的画质是否可达”。完整预训练 I2V 是最快降低这项不确定性的方式，但它本身不是项目创新，也不是从零训练能力的证明。

**受控 RAE baseline**回答“在可用预算内，换表示究竟产生什么影响”。它必须包含匹配的数据、条件、评估和足够透明的计算成本。只比较两个 encoder 名称而改变整个 decoder/生成器，不足以进行因果归因。

**研究贡献**回答“VGGT 在已有强生成器之上提供什么不可被普通视觉特征、相机输入或更多参数替代的收益”。这需要几何相关指标与画质同时成立。

这三个交付物可以共享评估与数据基础设施，但不能用第一个的好看样例掩盖第二个失败，也不能拿第三个的深度准确率替代视频质量。

### 5.2 优先步骤：完整预训练 I2V 参照

先在固定 32 个 held-out 首帧、固定 seeds 上运行完整原生 I2V。保留模型支持的分辨率、帧数与预处理，之后按相同时间段导出对照；不能为了兼容 R7 的 518×518、9 帧接口，先随意改原生模型。

主参照使用不含未来信息的固定通用 prompt 或空 prompt，并记录到底采用哪一种；不得使用从完整未来视频产生的 caption。另行报告首帧自动描述的增强参照时，必须标出新增的图像描述处理。结果包括原生完整视频与严格对齐的短片，避免只裁出最好的瞬间。

验收首先是定性的：建筑轮廓是否稳定、纹理是否重复、是否有持续而合理的运动。随后扩展到更多固定首帧。若完整原生模型也不工作，应先修复 NPU 推理、输入格式或 checkpoint 组合；此时不应训练任何新的几何模块。

这一步不需要训练 48 卡，也不依赖 R7 重新编码。需要先测 NPU 支持、内存与吞吐；不能根据 CUDA 示例承诺运行时间。已有 14B 适配出现过内存问题，不能默认全量训练可行。

拿到合理的原生视频后，可增加一个无需训练的关键对照：把它按当前窗口规则取帧，经过冻结 R7 的 encode/decode，再与同尺寸、同时间采样的原生结果并列。该对照不是重复已经完成的真实数据兼容性审计，而是检查 R7 能否承载一个已知合理的生成结果。

如果原生视频好、R7 重放也保持结构，而 R7 diffusion 很差，那么有用的未来视频确实能被当前表示/decoder 表达，主要缺口在如何学习与采样到这些表示；它仍不证明这个分布易学。如果原生视频好、经过 R7 就明显变坏，则应优先检查 codec 的感知质量和域适配。这个实验也不能冒充 R7 diffusion 成功：视频仍由原生模型生成。

### 5.3 保留 RAE 路线，但补齐可解释的对照

如果继续判断 R7 的潜力，优先补一个**原生 VAE 表示 + 同类从零 DiT**对照，用相同视频窗口、首帧信息、训练暴露次数和评估 seeds。应尽量匹配 backbone 宽深与计算预算，并完整报告 VAE/R7 不同的时空压缩、token 数、输入输出 head 参数及计算量，不能虚称严格单因素。

已有 R7 单域结果作为参照，不必原样再训一次。新的 VAE 控制应尽量沿用已审计的数据和 flow 组件，但不能把不兼容时空接口强行装进旧 tokenizer。6000 步可作为相同暴露量的观察点，不能继续作为未经证明的充分收敛标准。

| 新控制的结果 | 优先解释 | 后续投入 |
|---|---|---|
| VAE 控制能生成明显合理的视频，R7 不能 | 表示、条件和 decoder 接口成为主要嫌疑 | 研究新的 VGGT 生成表示，暂不扩大全域 scratch 训练 |
| 两者都差，而原生预训练 I2V 好 | 当前 scratch 数据/预算/生成器或共有实现不足 | 优先迁移完整生成先验；不因 R7 单臂失败换 encoder |
| 两者都可学习，R7 收敛明显更慢 | 表示效率问题更明确 | 做预算曲线和新的 tokenizer，而非一次性比较终点 |
| 原生 I2V 本身异常 | 生成基础设施或输入条件尚不可信 | 先修参照，不进入架构归因 |

如控制仍不足以归因，再考虑 DINO 表示，不同时铺开多个新 AE。每个新分支必须回答上一步无法回答的问题。

### 5.4 新 VGGT 表示的设计标准

不建议在现有 R7 上直接降到 32 通道、加球面约束或把 GroupNorm 换掉。新的表示应当单独训练、独立命名，并通过以下标准后才进入规模化 diffusion：

1. **条件与目标契约明确。** 首帧条件完全由首帧得到，预测时不得读取未来；如果目标依赖上下文，应将上下文定义固定。后续若做自回归，要验证前缀扩展/分块编码一致性。
2. **几何内容经过验证。** 测试压缩前后关系或几何可读性。深度任务头在这里是表示审计工具；如果压缩后几何优势消失，就不能继续用 VGGT 的名称承担论证。
3. **RGB 质量与生成稳健性同时评价。** 比较 RAW 重建和真实 diffusion 误差下的解码；只用各向同性随机扰动不足以模拟结构性预测误差。
4. **生成成本进入 tokenizer 选择。** 在相近重建/感知质量下比较下游学习曲线，而不是固定选 PSNR 最高的 AE。低维、低频或高语义各有代价，没有通用最优通道数。
5. **输入噪声匹配实际表示。** 测压缩后的通道尺度、token 幅度和方向约束；需要改变空间时重新定义训练路径与采样，不对旧 checkpoint 只改推理投影。

可先检验“单帧几何表示”和“整窗 R7”在相同帧上的差异，再决定是否重新引入 temporal bottleneck。去掉时序压缩会增加成本，所以它适合诊断或小规模原型，不自动是最终架构。

### 5.5 更有价值、但仍待验证的研究假设

建议把候选主张限定为：**首帧条件下的几何表示，应把可从输入继承的结构与需要随机生成的变化分开，使视频生成器将容量更多用于未知内容。** 外部轨迹不属于必要输入。

一种候选系统由三部分组成：首帧 appearance 编码；由首帧几何及噪声预测的紧凑未来结构状态；保持已有生成能力的条件视频 renderer。未来结构可包含相对运动、遮挡/可见性及低维空间关系，训练可以从完整真实片段获得目标，但推理只使用预测状态。

该分工类似已有两阶段生成，不能仅靠结构图宣称创新。真正要验证的是：预测状态是否比直接 feature 对齐更有用、是否降低达到同等画质的训练成本，以及收益是否来自几何而非添加一个额外网络。如果 renderer 忽略结构分支，或者换成同规模 DINO 特征同样有效，则不支持“VGGT 几何贡献”。

可以保留以下方法假设，按证据选择一个，不同时堆叠：

| 假设 | 关键验证 | 否定信号 |
|---|---|---|
| 压缩损失了 VGGT 有用关系 | 压缩前后关系读出，并对新表示做相同预算生成 | 几何已保留却生成仍同样差 |
| 上下文变化制造无用的预测目标漂移 | 同一真实帧在允许的不同历史下编码；分离真实运动与表征变化 | target 变化很小，或修正后无生成收益 |
| 首帧可继承信息重复建模浪费预算 | 绝对目标与条件表示在等质量/等预算下比较 | 只是复制画面、减少运动，或成本不降 |
| 实际生成误差与 decoder 训练分布不匹配 | 用训练集生成误差适配并保持干净重建参照 | 只能美容纹理，无法修复结构与时序 |

条件 tokenizer、残差建模和两阶段 renderer 都已有先例，创新应落在验证出的具体机制与相对强基线的优势。当前最诚实的定位是“有候选问题，尚未有可靠壁垒”。

## 6. 验证顺序与停止条件

### 6.1 建议的执行顺序

| 阶段 | 产物 | 进入下一阶段的条件 |
|---|---|---|
| A：原生 I2V，只推理 | 同首帧完整视频、短窗对照、运行配置和耗时；随后冻结 R7 重放生成视频 | 原生链路生成合理且可复现，并知道 codec 是否能保留其质量 |
| B：一个受控 scratch VAE 分支 | 与 R7 对照的学习曲线、同输入视频和资源记录 | 能区分表示问题与共有训练预算问题；若均差，先调整训练策略 |
| C：有界的新表示或完整先验适配 | 明确一个假设、一个主要改动、相同评估 | 同时有 RGB 质量与机制证据，而非只降 latent loss |
| D：扩展 HQ、几何贡献实验 | 多数据量曲线、跨来源评估、几何消融 | 小规模结果稳定，新增数据仍带来可测增益 |

这是决策顺序，不是本轮已授权并已完成的实验。A 优先，B 用于对 RAE 主张做必要归因；若工程目标先求高质量，则可先沿完整先验做适配，同时不把它冒充 scratch RAE 成功。

禁止默认再开 aux 层数、shift、Euler 步数或早停网格。只有新证据明确指向其中一项，才重新打开对应问题。已有 memory 和轨迹输出足以支持这一停止决定。

### 6.2 评价必须同时看质量、运动和几何

近期保留 32 视频 × 2 seed 作为快速诊断，后续至少扩展到数百个独立片段形成可靠趋势。报告每视频分布和置信区间；32 个视频不能支持稳定的分布距离结论。FVD 等指标只有在样本量、时长、帧率和评估实现一致时才有可比性，不能跨设置照抄论文数值。

画质验收包含盲评偏好、结构变形率、纹理稳定性和固定失败案例，paired L1/PSNR 只作为补充。必须同时报告运动强度、近静止率与不同 seed 的差异，防止通过冻结画面获得漂亮的一致性分数。未来路径不唯一，不应把偏离记录轨迹的合理视频全部计作失败。

几何验收应使用至少一种独立于训练 VGGT 的估计或跨帧匹配证据，避免只测“是否更像教师”。深度重投影误差需要遮挡、动态区域和尺度处理；没有尺度真值时不能声称米制几何精确。所有指标都保留对 RAW 视频的同流程参考，以估计评价器自身误差。

几何方法至少比较：原生模型、同预算的普通视觉特征条件、几何条件、打乱/移除几何关系。若后来开放相机控制，再增加 camera-only 与 geometry+camera 对照。不能让被比较的方法分别拥有不同的未来信息。

预先规定工程选择原则：改动在重复 seeds 上有清晰画质收益，或在画质基本不降时有稳定几何提升，才继续扩展。若没有明显收益、只是配对 L1 的小数点改善，则停止该分支。这些是选择规则，不是对成功率的承诺。

### 6.3 关于算力、记录与恢复

48×910B 是可用资源，不是所有试验都应占满的理由。先测原生推理和一个训练 step 的显存/吞吐，再定模型规模。官方 GPU 速度无法直接换算为 NPU 时长，也不能把旧 178M 的 step time 外推到 14B。

未来 launcher 继续沿用已验证的双读双写、OBS/平台回退、完整训练状态恢复、独立 namespace、逐步 checkpoint、RAW/AE/生成样例、节点与 rank 完成回执。新 representation 禁止复用旧 normalization/cache/optimizer 状态而不更换契约。

性能日志需明确 DI throughput 的单位，同时报告全局 windows/s、frames/s、优化器 steps/s、数据读取等待、有效训练时间及包含缓存/评估/同步的总墙钟时间。保留 sample exposure 与 unique video/window 数；二者不可混用。改变架构后不承诺与之前相同训练时间。

## 7. 文献适用范围补充

下表用于限定可借鉴的思想。核心路线依据前文的方法和实验设置；其他条目只用于定位相关性，不据此提出已验证的训练配方。

| 工作 | 可借鉴内容 | 不应据此下的结论 |
|---|---|---|
| REPA | 在生成模型内部提供表示学习监督[^28] | 任意 clean-latent aux 就能等价复现 |
| REPA-E | 联合优化 tokenizer 与 DiT 的方向[^29] | 在尚未有成功 baseline 时直接端到端训练就更稳 |
| Reconstruction vs. Generation / VA-VAE | 重建与可生成性需要共同设计[^30] | 当前 PSNR 高的 AE 必然应当生成得好 |
| REGLUE | 全局与局部语义对 latent 结构的作用[^31] | 几何与外观的简单拼接自动具备所需关系 |
| Frequency Perspective | RGB 频率和 latent 空间频率不能直接等同[^32] | 未完成的 subspace 实验足以支持删高频/PCA |
| Back to the Features | 利用冻结视觉表示建立视频世界模型[^33] | 感知预测成绩等于高质量 RGB |
| 4DLangVGGT、VGGT-DP | 几何表示用于语言场或机器人任务[^34][^35] | 任务头可读出就证明生成器可采样 |
| Vision Transformers Need More Than Registers | 表示与 token 设计的相关证据[^36] | 当前 R7 失败来自 registers，必须改 backbone |
| Cubic Discrete Diffusion | 高维表示上的离散生成替代方向[^37] | 换离散目标即可解决未定位的时序与数据问题 |
| VFM as Visual Tokenizers for AR | foundation 表示与自回归图像生成可结合[^38] | 当前 legacy 时序 codec 可直接改 AR |
| Language-Guided Image Tokenization | 把条件信息纳入 tokenization，减少重复编码[^39] | 给当前首帧任务使用真实未来 caption 是合法对照 |
| Conditioned Diffusion-based Tokenizer | 将条件 diffusion 用作视频解码器的已有方向[^40] | 更强 decoder 能自动恢复任意错误的生成结构 |

## 8. 最终建议

当前 R7 配方应作为完整的失败基线保存，而非继续充当所有新实验的固定前提。保留首帧任务、保留 VGGT 研究目标，同时重新取得两个锚点：完整生成先验在本数据上的实际质量，以及可解释的 VAE/R7 训练对照。

最优先缺失的不是又一个 loss，而是一个成功控制。没有它，“数据不够”“模型太小”“归一化有问题”都容易变成不可证伪的解释。拿到控制后，才值得围绕几何信息保留、上下文一致性、条件压缩或 decoder 误差适配选择一个主要机制。

若最终需要完整预训练 renderer 才能获得质量，应明确承认这一点，并用严格消融证明 VGGT 降低了生成成本或改善了几何一致性。若一个经过重新设计的 VGGT RAE 在匹配预算下成功，则贡献应围绕让几何表示可生成的具体机制展开。两条路径都有研究价值，但目前没有证据保证哪条会成功。

## Sources

[^1]: 本项目，[DOMAIN_SINGLE_REVIEWED_RESULTS.md](./DOMAIN_SINGLE_REVIEWED_RESULTS.md)，2026-09-15；6000 步完成、评估及成本记录。
[^2]: 本项目，[DOMAIN_TRAJECTORY_RESULTS.md](./DOMAIN_TRAJECTORY_RESULTS.md)，2026-09-15；冻结 EMA6000 的完整轨迹和可视证据边界。
[^3]: 本项目，[DOMAIN_AE_REPLAY_RESULTS.md](./DOMAIN_AE_REPLAY_RESULTS.md)、[DOMAIN_UNIFORM_RUNBOOK.md](./DOMAIN_UNIFORM_RUNBOOK.md)；AE 重放、数据契约及分域设计。
[^4]: Wan-Video，[Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)，访问 2026-09-15；模型清单及 I2V 推理接口。
[^5]: Hu et al.，[Multiplayer Interactive World Models with Representation Autoencoders](https://arxiv.org/pdf/2607.05352v2)，2026-07，v2；任务、规模和 scaling 实验。
[^6]: MIRA 团队，[raev2_codec_tdown.yaml](https://github.com/mira-wm/mira/blob/main/configs/model/raev2_codec_tdown.yaml)，访问 2026-09-15；公开 codec 配置。
[^7]: Xie et al.，[VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders](https://arxiv.org/html/2607.14088v2)，2026-07，v2；§4.1 的受控与预训练适配设置。
[^8]: Guo, Wu, Fei，[V-RAE: Rethinking Video Latent Spaces for Generation](https://arxiv.org/html/2608.13556v1)，2026-08；方法及附录训练/数据设置。
[^9]: Zheng et al.，[Diffusion Transformers with Representation Autoencoders](https://arxiv.org/html/2510.11690v1)，2025-10；原始 RAE 方法。
[^10]: 论文作者，[Improved Baselines with Representation Autoencoders](https://arxiv.org/html/2605.18324v2)，2026-05，v2；§2 和附录 A，归一化及内部监督。
[^11]: Zhu et al.，[Beyond the Last Layer: Multi-Layer Representation Fusion for Visual Tokenization](https://arxiv.org/html/2605.10780v1)，2026-05；融合与生成兼容性。
[^12]: 论文作者，[Latent Diffusion Model without Variational Autoencoder](https://arxiv.org/html/2510.15301v1)，2025-10；§3.3、Table 4，SVG 分布对齐。
[^13]: Weijler et al.，[Latent Riemannian Flow Matching for Geometry-Grounded 3D Foundation Models](https://arxiv.org/html/2607.19120v1)，2026-07；§3.2–4、Table 5、Appendix B。
[^14]: 论文作者，[Repurposing Geometric Foundation Models for Multi-view Diffusion](https://arxiv.org/html/2603.22275v1)，2026-03；§4.3–4.4、Appendix C.1。
[^15]: 论文作者，[Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction](https://arxiv.org/html/2601.04090v1)，2026-01；生成/解码结构。
[^16]: 论文作者，[VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model](https://arxiv.org/html/2603.12655v1)，2026-03；§3 的几何预测任务。
[^17]: 论文作者，[Flow Matching in Feature Space for Stochastic World Modeling](https://arxiv.org/html/2606.29059v1)，2026-06；§3.1–3.2，下游感知评估。
[^18]: 论文作者，[Geometry Forcing 官方项目页](https://geometryforcing.github.io/)，论文 2025，访问 2026-09-15。
[^19]: 论文作者，[CamGeo](https://arxiv.org/html/2605.30895v1)，2026-05；相机条件与几何先验。
[^20]: 论文作者，[SemanticGen: Video Generation in Semantic Space](https://arxiv.org/html/2512.20619v1)，2025-12；§3，语义生成与预训练 VAE renderer。
[^21]: Li et al.，[VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_VMem_Consistent_Interactive_Video_Scene_Generation_with_Surfel-Indexed_View_Memory_ICCV_2025_paper.pdf)，ICCV 2025。
[^22]: 论文作者，[Retrieve What's Missing: Coverage-Maximizing Retrieval for Consistent Long Video Generation](https://arxiv.org/html/2606.02479v1)，2026-06；CovRAG 方法。
[^23]: Shen et al.，[Lyra 2.0: Explorable Generative 3D Worlds](https://arxiv.org/html/2604.13036v1)，2026-04；§4.2–4.3。
[^24]: 论文作者，[TourPhysics: Bringing Physics to World Models for Exploration and Manipulation from a Single Image](https://arxiv.org/html/2609.04911v1)，2026-09；置信度残差与原生路径回退。
[^25]: 论文作者，[VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward](https://arxiv.org/html/2603.26599v1)，2026-03。
[^26]: 论文作者，[Geometric Autoencoder for Diffusion Models](https://arxiv.org/html/2603.10365v1)，2026-03；latent 几何、语义与稳健重建。
[^27]: SpatialVID 团队，[SpatialVID: A Large-Scale Video Dataset with Spatial Annotations](https://arxiv.org/html/2509.09676v1)，2025-09；数据与标注流程。本项目对象数与分域数来自本地已完成清单审计，不是论文原始统计。
[^28]: Yu et al.，[Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940)，2024-10。
[^29]: 论文作者，[REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](https://arxiv.org/abs/2504.10483)，2025-04。
[^30]: 论文作者，[Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://arxiv.org/html/2501.01423v1)，2025-01。
[^31]: 论文作者，[REGLUE Your Latents with Global and Local Semantics for Entangled Diffusion](https://arxiv.org/abs/2512.16636)，2025-12。
[^32]: 论文作者，[Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective](https://arxiv.org/html/2511.22249v1)，2025-11；§3 区分 RGB 与 latent 频率。
[^33]: 论文作者，[Back to the Features: DINO as a Foundation for Video World Models](https://arxiv.org/abs/2507.19468)，2025-07。
[^34]: Wu et al.，[4DLangVGGT: 4D Language-Visual Geometry Grounded Transformer](https://arxiv.org/abs/2512.05060)，2025-12。
[^35]: 论文作者，[VGGT-DP: Generalizable Robot Control via Vision Foundation Models](https://arxiv.org/abs/2509.18778)，2025-09。
[^36]: 论文作者，[Vision Transformers Need More Than Registers](https://arxiv.org/abs/2602.22394)，2026-02。
[^37]: 论文作者，[Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](https://arxiv.org/abs/2603.19232)，2026-03。
[^38]: 论文作者，[Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation](https://arxiv.org/abs/2507.08441)，2025-07。
[^39]: Zha et al.，[Language-Guided Image Tokenization for Generation](https://arxiv.org/abs/2412.05796)，2024-12，修订 2025-04。
[^40]: 论文作者，[Rethinking Video Tokenization: A Conditioned Diffusion-based Approach](https://arxiv.org/abs/2503.03708)，2025-03。
