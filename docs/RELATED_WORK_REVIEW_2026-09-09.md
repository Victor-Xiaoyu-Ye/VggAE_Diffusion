# 几何感知视频 Diffusion：相关工作与项目决策

检索日期：2026-09-09。面向当前首帧条件、9帧联合生成、冻结 R7 AE、SpatialVID 子集、48×910B 的实际任务。

## 结论先行

**让正在训练的 shift3 完成受控比较。接下来最有价值的工作不是继续搜索一个万能 shift，而是区分“预测 latent 本身不合理”与“解码器放大预测偏差”，然后验证轻量的中间层 clean-latent 监督。预训练迁移则应研究接口对齐，不能再假设随机换输入输出头后 Wan 的能力自然保留。**

本轮为广泛定向检索及重点原文核读，不是可证明穷尽的全数据库系统综述。覆盖下表32项论文/资源；重点阅读 RAEv2、LV-RAE、DC-VideoGen、GLD、Gen3R、VideoRAE、V-RAE、GeoFlow 的方法部分，核对部分官方仓库发布状态。未运行外部模型，不把公开代码等同于910B可运行。二手聚合页仅用于发现线索，结论引用作者论文、项目页、官方仓库或会议页面。

## 1. 先把我们已经知道的事情说准确

本项目实测：legacy AE 回放24.49dB；EMA6000自由生成仍失真；Euler128仅改善约1.21%；首帧置零/打乱显著变差；训练子集预览也不好。shift3当前正在运行，尚无结果。

这些证据排除了旧归一化语义错误，并降低了采样积分误差、条件完全未生效作为主因的可能性。**它们没有排除解码器对非编码器输出的敏感性。**“AE重建通过”只检验了 D(E(x))，还没有检验 D(z+δ)。我此前把瓶颈概括为生成器学习，范围过窄，应修正为“生成分布学习与解码误差敏感性尚待拆分”。

当前 R7 是多级压缩并混合后的C192，不是未经改造的1024/2048维VGGT tokens；主干宽度768也已经大于单token维度192。不能直接用原始RAE的宽度不足论证强制扩大模型。geo/tex经codec混合，不能把最终前112通道解释为纯几何。

## 2. 最值得借鉴的六组思想

### A. RAEv2 / iREPA：使用基础模型latent，不等于去噪中间层已学会结构

RAEv2发现RAE与中间层表征监督可以互补，并把浅层clean预测作为内部引导的基线；其导航实验有多帧历史和动作条件，不能直接当作我们的首帧I2V结果。iREPA强调局部空间关系与投影/归一化设计。[RAEv2原文](https://arxiv.org/html/2605.18324v2)、[iREPA会议论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html)。

**本项目推断与候选实验：**冻结现有AE，在DiT中部增加轻量head预测已有clean R7 target；先只比较辅助训练的效果，采样仍用最终head、guidance=0。这样无需重新抽VGGT teacher、不改缓存，能验证中间层直接监督是否帮助学习。它是压缩latent的深监督对照，不能称为已保留原始VGGT几何，也不是新颖点。不要同时加入卷积投影、内部guidance和新loss权重，否则无法归因。具体head位置及权重须在实施前定义，不能照抄别人的第8层。

### B. LV-RAE / 原始RAE：重建好，仍可能把生成偏差放大成破碎纹理

LV-RAE讨论语义与细节兼具的latent上，decoder沿数据流形外方向放大误差，并研究噪声增强的鲁棒解码。原始RAE同样使用decoder噪声增强；但Scaling RAE在更大规模实验中发现该技巧收益很小，说明它不是通用必选项。[LV-RAE](https://arxiv.org/html/2602.08620v1)、[RAE](https://rae-dit.github.io/)、[Scaling RAE](https://arxiv.org/abs/2601.16208)。

**最优先的低成本检查：**不训练任何新AE，保持原anchor，以真实future latent z和已有生成 z_gen 做：

1. 定向插值：D(z+α(z_gen−z))，α取0、0.05、0.1、0.25、0.5、1。
2. 等RMS随机扰动：在同一归一化空间按每个时间slot校准噪声幅度，再逆归一化解码。
3. 比较RGB/感知误差增长、逐帧结构破坏、不同方向的敏感性；不能把随机方向自动叫作严格的法向方向。

若很小扰动就导致明显形变，先考虑仅decoder鲁棒微调的独立副本；若只有接近完整生成误差时才坏，优先继续改生成分布。微调前必须证明敏感性问题，保留原AE与干净重建对照；不重新训练整个encoder、也不覆盖已通过的checkpoint。

### C. DC-VideoGen / Gen3R：迁移先验要明确对齐什么

DC-VideoGen先对齐新patch embedder到原生embedding，再冻结DiT适配输入输出接口，最后LoRA；它报告直接随机换接口微调会不稳定。Gen3R把几何latent与原生外观latent对齐后联合生成，两种对齐发生在不同层面。[DC-VideoGen方法](https://arxiv.org/html/2509.25182v1)、[Gen3R方法](https://arxiv.org/html/2601.04090v1)。

**本项目推断：**如果scratch分支仍无法获得合格视频，Wan路线应先在原生表示/任务接口上建立质量参照，再做“同一批视频双编码→接口对齐→冻结主干阶段→有限微调”。R7的18×18、t2布局与Wan不同，空间pooling、物理时间对应、加噪后的embedding对应均须验证；只对齐clean特征不自动证明整条噪声路径兼容。这里不能照搬H100工时预测910B成本。

发布状态需要纠正：本次打开DC-VideoGen官方仓库，仍主要是README和图，明确写代码/权重待发布；有论文和仓库链接并不等于有完整可运行实现。[官方仓库](https://github.com/dc-ai-projects/DC-VideoGen)。

### D. V-RAE / VideoRAE：借鉴评估和监督位置，不照搬任务

V-RAE的tFVD使用时间邻居插值后的解码结果测试latent局部时间平滑性；它也用辅助clean预测分支。VideoRAE的REPA主要施加在AE的decoder中间特征，和去噪器的中间监督不是同一个位置。[V-RAE](https://arxiv.org/html/2608.13556v1)、[VideoRAE](https://arxiv.org/html/2607.14088v2)。

**本项目推断：**可以借鉴时间邻居插值压力测试，但我们的独立首帧与t2 future chunk不是同质时间token，先只在future内部插值并保持长度及anchor。9帧、16个视频上的简化实验不能冒充官方tFVD协议或正式FVD结论。它用于发现失真敏感性，而不是把PSNR25再换成另一个单一门槛。

### E. Geometry Forcing / VideoREPA / VideoWeave：差异性必须更具体

Geometry Forcing用几何特征方向与尺度监督；VideoREPA蒸馏跨时空token关系；VideoWeave已把隐式几何latent与视频latent放入联合去噪空间，并构建配对数据。简单的“加VGGT监督”“双流”“关系loss”都已有近邻。[Geometry Forcing](https://geometryforcing.github.io/)、[VideoREPA](https://arxiv.org/abs/2505.23656)、[VideoWeave](https://arxiv.org/abs/2606.14162)。

**本项目可研究的假设：**压缩和去噪过程对跨帧表面对应关系的破坏，是否解释生成结构漂移；针对该机制的设计是否优于普通latent深监督、semantic-REPA和geometry-REPA。这里强调可证伪实验，而非宣称首次。仅使用R7作为目标时，必须另行证实压缩后仍能恢复几何关系；否则只是一个带几何来源的外观表示。

### F. GeoFlow / GeCo：评价应区分相机运动与独立物体运动

GeoFlow结合几何诱导运动与动态外观对应来评价生成一致性，再做RL微调；GeCo主要面向静态场景的形变/遮挡一致性。两者都不能被简化为“帧差越小越好”。[GeoFlow](https://arxiv.org/html/2605.18365v1)、[GeCo](https://geco-geoconsistency.github.io/)。

**本项目推断：**优先借评估思想，不现在上RL/DPO。静态背景测相机诱导flow与观测flow的一致性，动态区域测沿轨迹的身份/外观保持；记录遮挡、置信度和失败率，并与RAW、AE重建、首帧重复一同评估。运动退化到静止不能被奖励为几何优秀。最好用独立于训练teacher的评估器，避免只测“像不像VGGT自己的特征”。

## 3. 32项工作/资源地图

下表是筛选地图，不是32项完整复现；“可借鉴”都是针对本项目的判断。日期以所链论文/资源为准，不用搜索引擎相对时间判断优先权。

| # | 工作与一手入口 | 解决的问题 | 对本项目的取舍 |
|---|---|---|---|
| 1 | [RAE](https://arxiv.org/abs/2510.11690) | 表征latent上的图像扩散 | 噪声、容量、decoder鲁棒性分开审计 |
| 2 | [Scaling RAE](https://arxiv.org/abs/2601.16208) | 扩大T2I规模后配方是否仍成立 | 支持噪声调度对照，不支持宽head必需论 |
| 3 | [RAEv2](https://arxiv.org/abs/2605.18324) | 表示聚合、深监督、内部引导 | 高优先级思想；不换现有encoder |
| 4 | [iREPA](https://arxiv.org/abs/2512.10794) | 对齐中空间关系的重要性 | 中间监督后的投影设计候选 |
| 5 | [LV-RAE](https://arxiv.org/abs/2602.08620) | 细节与decoder扰动敏感性 | 优先做冻结压力测试 |
| 6 | [JiT](https://arxiv.org/abs/2511.13720) | 直接预测clean数据 | 当前已用x0，不能当成未尝试的新修复 |
| 7 | [VideoRAE](https://arxiv.org/abs/2607.14088) | 冻结视频编码器到生成latent | AE监督位置、压缩参考；不同任务 |
| 8 | [V-RAE](https://arxiv.org/abs/2608.13556) | 视频表征AE与未来预测 | 插值诊断、辅助head；不复制Cityscapes配方 |
| 9 | [VGGT-World](https://arxiv.org/abs/2603.12655) | 几何feature自回归预测 | clean目标可参考，但不生成RGB，不能当视频成功基线 |
| 10 | [MIRA](https://arxiv.org/abs/2607.05352) | RAE多人游戏世界模型 | 规模与codec消融参考；游戏/动作条件不同 |
| 11 | [GLD](https://arxiv.org/html/2603.22275v1) | 几何feature空间多视图NVS | 多层feature一致性；给定相机、主模型DA3，非自由动态视频 |
| 12 | [Gen3R](https://arxiv.org/abs/2601.04090) | 几何与外观联合场景生成 | 对齐思想；保留原生外观latent，与纯R7路线不同 |
| 13 | [DC-VideoGen](https://arxiv.org/abs/2509.25182) | 视频模型迁移到新AE | 最相关的接口迁移方法参考，代码未完整发布 |
| 14 | [Geometry Forcing](https://arxiv.org/abs/2507.07982) | 中间层几何方向/尺度对齐 | 必须考虑的几何监督基线 |
| 15 | [VideoREPA](https://arxiv.org/abs/2505.23656) | 视频关系蒸馏 | 比单帧cosine更贴近时间关系；非新颖点 |
| 16 | [CREPA](https://arxiv.org/abs/2506.09229) | 跨帧表征对齐微调 | 邻帧约束参考，但不等于3D对应 |
| 17 | [VideoWeave](https://arxiv.org/abs/2606.14162) | 联合几何/视频latent后训练 | 双流路线的重要直接近邻 |
| 18 | [GeoFlow](https://arxiv.org/abs/2605.18365) | 动静分解的几何一致性奖励 | 先借评估，暂不做RL |
| 19 | [VideoGPA](https://arxiv.org/abs/2601.23286) | 几何偏好DPO | 后训练路线近邻；当前基础质量不足 |
| 20 | [WorldReel](https://arxiv.org/abs/2512.07821) | RGB、pointmap、camera、flow联合4D | 动态几何建模参考；监督/数据成本较高 |
| 21 | [WorldWarp](https://arxiv.org/abs/2512.19678) | 3D缓存warp与区域噪声调度 | 可见/新区域分工；更偏相机控制与长序列 |
| 22 | [RayPE](https://raype-project.github.io/) | 射线位置编码进入attention | 需要相机射线；不能偷偷用未来GT姿态 |
| 23 | [RoGe](https://arxiv.org/abs/2609.02847) | 隐式重建与NVS端到端耦合 | 近期NVS近邻，不能与无轨迹I2V混比 |
| 24 | [GeoNeXt](https://arxiv.org/abs/2608.28549) | 视频生成先验用于几何估计 | 方向相反，只作先验迁移参考 |
| 25 | [Matrix-Game 3.5](https://arxiv.org/abs/2608.29910) | patch记忆的流式交互世界模型 | 长程记忆后续参考，不解决当前9帧失真 |
| 26 | [History-Guided Video Diffusion](https://arxiv.org/abs/2502.06764) | 历史条件与引导 | 条件guidance需专门训练；不直接用置零支路做CFG |
| 27 | [GeCo](https://geco-geoconsistency.github.io/) | 几何形变与遮挡评估 | 用于静态背景并防止静止作弊 |
| 28 | [VBench-2.0](https://arxiv.org/abs/2503.21755) | 视频内在真实性多维评估 | 几何分数应与画质/运动多维联合报告 |
| 29 | [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) | 几何标注真实视频数据 | 利用动静/相机/场景字段分层，先核对本地镜像 |
| 30 | [MeanFlow+RAE](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_MeanFlow_Transformers_with_Representation_Autoencoders_CVPR_2026_paper.html) | 少步生成的稳定训练 | 依赖teacher等训练设计，当前不优先改少步目标 |
| 31 | [DreamWorld](https://github.com/ABU121111/DreamWorld) | 多种世界知识联合建模 | 多teacher并非空白方向，暂不增加在线teacher负担 |
| 32 | [GimbalDiffusion](https://arxiv.org/abs/2512.09112) | 重力/相机控制及SpatialVID再平衡 | 数据与相机评估借鉴，任务应单列 |

## 4. 数据形式对借鉴方法的实际限制

SpatialVID公开资源包含camera、动态mask、caption和运动字段，但部分depth为可选文件，标注帧通常按int(fps/5)抽取。我们现在的9帧/1秒取样不能直接拿第i个标注当第i个RGB；若有resize/crop，还要同步调整intrinsics。[官方数据说明](https://huggingface.co/datasets/SpatialVID/SpatialVID)。

本项目建议：先盘点已下载镜像里真正存在的标注及原视频来源；按源视频分组划分，按静态背景/动态物体、相机移动/静止、遮挡等分层评估。扩大数据时优先增加独立视频和有效时间窗口，避免将9935条固定窗口反复训练的曝光次数当成新数据量。

几何teacher可在训练时看完整视频作监督；但当前任务推理只有首帧，不能引入真实未来深度、flow或相机轨迹作为条件。如果改为相机控制，须另设任务及公平基线，不把额外条件带来的质量提升算作表示本身优势。

## 5. 结合48×910B的实验顺序

| 优先级 | 工作 | 是否改变当前训练 | 可回答的问题 |
|---|---|---|---|
| P0 | 完成shift1/shift3同预算对照 | 不改正在跑的配置 | 高噪声训练分配是否有帮助 |
| P0 | 冻结decoder、已有生成误差方向与随机方向压力测试 | 不训练；独立输出 | 解码器是否放大小偏差 |
| P1 | 单一中间clean-R7 head，guidance先为0 | 新实验，复用AE/cache | 深监督能否改善训练集和留出自由生成 |
| P1 | RAW/AE/生成/静止copy的动静分层评估 | 离线评估 | 画质与几何收益是否真实 |
| P2 | 仅在压力测试失败时微调decoder副本 | 不覆盖原AE | 鲁棒性与干净重建的权衡 |
| P2 | 原生视频质量基线与接口对齐迁移 | 独立路线 | 能否继承视频先验而非从头学全部动态 |
| 暂缓 | RL/DPO、多teacher在线训练、长rollout、3DGS缓存、少步蒸馏 | 不实施 | 避免同时引入多个尚未验证的问题 |

不从论文GPU时间推算910B时间；所有新增组件先验证算子、实际显存、DI吞吐、双写及断点恢复。中间监督若使用缓存R7不增加VGGT在线编码成本；原始几何teacher监督则需要新sidecar及时间对齐契约，两者不是同一个实验。

## 6. 差异性应怎样建立

建议研究命题：**几何关系在压缩、去噪、RGB解码三阶段如何被保留或破坏，以及针对最主要破坏环节的设计能否在动态真实视频中带来收益。** 这是研究假设，尚非项目结论或新颖性保证。

应先定位：原始几何teacher→压缩R7→真实latent解码→生成latent→生成RGB。能直接读feature时做feature关系probe；其余阶段对RGB用独立几何/跟踪器评估，注明重新估计带来的误差，不能把两者当同一测量。

需要的对照至少包括：普通R7-DiT；同预算latent深监督；有条件时semantic教师与geometry教师对照；有质量优先路线时原生VAE普通微调与几何增强微调。动静区域分开报告。只有方法对准了可复现的失效机制，并超过这些近邻方法，才能将收益归因于几何设计。

**当前没有证据表明仅把VGGT换进AE、把geo/tex拼接、增加一个cosine/关系loss，便足以建立论文贡献。** 也没有哪篇论文可以保证PSNR25的AE配上通用diffusion配方就一定成功。

## 7. 交付与边界

本轮仅检索、核读和更新研究判断；没有更改正在训练的stage33，没有实现上述P0/P1新实验。报告所称“已发布代码”仅为页面检查，未验证完整训练可复现。RAEv2官方仓库可见stage1/stage2训练入口；DC-VideoGen仍待完整发布；VideoWeave本次项目页未找到训练代码入口，不能视为可直接复现。[RAEv2仓库](https://github.com/nanovisionx/RAEv2)、[DC-VideoGen仓库](https://github.com/dc-ai-projects/DC-VideoGen)、[VideoWeave项目页](https://videoweave.github.io/)。
