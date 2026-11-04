# 核心方法设计：用可解码表示传输编辑的可见效果

状态：2026-11-04 的研究设计，**尚未实现或验证有效**。不是 Stage61 现有的
辅助 loss，也不覆盖其启动配置。Stage61 继续作为正确对齐的监督基线。
本文给出一个可以先做小实验推翻的核心算法，不承诺方法新颖性已经确立。

后续实现更新：第一阶段冻结codec探针已落实为Stage62，运行方式见
`docs/EDIT_TRANSPORT_PROBE_RUNBOOK.md`。它实现受控2D对应/RGB效果的机制检验，
尚未实现本设计的自然对应、文本前端或真实native去噪写回；不依赖Stage61结果。

## 1. 选择的任务与 RAE 的角色

主任务：输入完整的固定窗口源视频 X 与文本 c，输出外观修改后的视频 Y。
保留源视频的动作、相机运动、几何边界和遮挡顺序。首轮聚焦颜色及近似漫反射
外观改变；几何变形、物体运动重定向、镜面材质、透明化和重光照不套用下面的
外观不变式。后续必须加入纹理类任务，不能只做容易的全局调色。

RAE 在本方法中是**可读写的视频编辑状态**，不是训练时旁接的特征教师。
预训练视频模型仍承担画质、细节和生成先验。RAE 提供“给定想要的可见改动，
应怎样改变表示”的操作接口；最终由这个操作修改真实 native 去噪状态。

直接 VGGT/DINO 监督并不先天地更差。额外 decoder 不增加输入信息，也不
保证语义、几何、外观自动解耦。VGGT/DINO 加上合适 decoder 后也属于可解码
表示这一范式。我们的 C256 是 128 Stream 分支 + 128 RGB 分支，分支名称
不证明物理解耦；本方法允许修改全部通道，不硬编码“只改 texture128”。

## 2. 一个明确可检验的 insight

**跨帧应该保持一致的是编辑的可见效果；同一 latent 改变量经不同源状态的
非线性 decoder，未必产生同一效果。** 现有 encoder 可以产生一致或相近
特征，但没有直接给出可见编辑效果的写回映射。

记源视频编码状态为 z_i=E_R(X)_i，固定 decoder 为 D。源 encoder 一次读取
完整可用窗口，不把本项目已经验证较差的单帧 reset 编码当作默认来源。
给第 i 帧的表示增加 delta_i，产生的可见改变是：

    r_i(delta_i) = D(z_i + delta_i) - D(z_i).

设 W_ij 是源视频可见表面从 i 到 j 的 RGB/像素对应，T_ij 是直接 feature
搬运方法给出的 token 对应。直接搬 token 隐含要求：

    r_j(T_ij delta_i) ≈ W_ij r_i(delta_i).

这不是重建能力的推论。对小改动进行一阶展开，所需条件为：

    J_D(z_j) T_ij ≈ W_ij J_D(z_i).

即使 E/D 的重建不错，这个关系也未必成立。D 非线性、源状态不同、token
空间分辨率低以及全局上下文，均可能影响它。这里的 J 是图像 decoder 的
局部响应，不是相机位姿、3D Jacobian 或已验证的几何头。

**可证伪命题 H1：** 在当前生产 AE 上，正确坐标与相同编辑强度下，直接
token 搬运存在显著的上述“不交换误差”；按目标帧 decoder 的局部响应校准
后，该误差减小，且独立帧上的目标编辑仍被执行。

如果实测不交换误差很小、或简单归一化/线性适配已解决，这条核心动机不成立。
不能把一般非线性数学事实冒充本项目已经观察到的失败原因。

H1也不能单独证明需要RAE：如果对应完全正确、表面可见且外观假设精确，
直接在RGB效果上warp就是强基准，甚至是该设置下的oracle。不能人为先把
编辑搬进错误的latent坐标，再以修复它来宣称优于像素方法。
真正有应用价值的第二个命题H2是：**对应稀疏、不确定或部分缺失时，目标源
状态的decoder可达编辑空间，是否比像素平滑/原生VAE提供更好的编辑补全与
错误抑制，并改善最终native输出。** 这来自decoder作为学习到的先验的潜在
作用，不由可解码性本身推出。需控制删边比例和对应扰动，用独立GT评测。
如果H1成立但H2不成立，只有解释性观察，不能据此重启大规模训练。

## 3. 可见效果的定义，避免把亮度一起锁死

RGB 差异 r 是必须保留的简单基线。主候选使用有界 log-ratio：

    R_i(delta_i) = log(D(z_i+delta_i)+epsilon) - log(D(z_i)+epsilon).

若源和编辑后对应同一漫反射表面，X=s*a、Y=s*a'，光照 s 没变，则理想情况下
log(Y)-log(X)=log(a')-log(a)，消除了同一表面随时间/视角变化的乘性阴影。
这只是限制条件下的动机，不是已分离 albedo，也不适用于镜面/透明/光照编辑。
epsilon 初值1/255；暗区及饱和区降低可靠性，log-ratio 做有界裁剪并报告占比。
RGB 差异 vs log-ratio 必须独立消融，不能把后者收益混算成 RAE 收益。

记 log 版本的局部响应为 A_i=diag(1/(D(z_i)+epsilon))*J_D(z_i)。校准后的
编辑传输在一阶近似下是一个正则化逆问题：

    delta_j* = argmin_delta ||V_ij^(1/2) (A_j delta - W_ij A_i delta_i)||²
                               + beta ||Lambda_j delta||².

V 是可见性/可靠性权重，Lambda 使用同一训练集固定的 latent 尺度归一化。
不能求高维完整 Jacobian/pseudoinverse；用 JVP/VJP 的矩阵自由迭代或直接对
delta 的非线性小步优化。最初不依赖 NPU forward-mode AD，先用已有 decoder
反传做 Gauss-Newton/梯度迭代的可用性验证。小步信赖域抑制远离 AE 数据流形。

这个式子是“把同样的可见编辑写进另一帧状态”的算法，而不是距离特征更近的
新命名。仅把 feature error 乘 J^T J 会退化成 decoded RGB loss，不是本方法。

## 4. 主架构和条件来源

文字版数据流：

    源视频 X + 文本 c -> 共享的预训练编辑前端 -> 编辑锚帧 K'
    源视频 X          -> 冻结 Stream/RAE       -> 各帧源状态 z_i
    源视频 X          -> 共享对应模块           -> W_ij、visibility
    z_i + K' + 对应    -> decoder 校准传输算子  -> 各帧编辑状态 delta_i
    delta_i + z_i     -> decoder              -> 可见编辑效果 R_i
    X + R_i          -> 原分辨率编辑参考       -> native VACE 实际去噪写回

文本完整版的锚帧由模型生成，用户不必提供。首选固定、已有的强单帧编辑器，
例如 [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)；
这是其图像编辑能力的复用，不声称当前最强或910B已经适配。
所有传播基线使用同一份锚帧缓存、相同编辑器预算、prompt 和 seed。
先用已经生成的 VACE/SFT 锚帧及清楚标注的 oracle-anchor 诊断，验证本算子，
不为一个未验证的传播机制先部署第二个大模型。

默认锚帧固定为首帧；1/3锚帧为独立消融，不能用 target 挑最好的时刻。
若首帧未见目标对象，应由预先定义的 source-only 选择规则处理并记录，不能
读取 edited target 决定。多锚帧需要共享外观参考，否则独立编辑本身也会不一致。

完整论文仍保留纯文本主协议。可选点击/mask 或用户编辑帧构成独立条件协议；
所有方法同条件比较。数据集 edited target 只能训练/评估，不能当作主协议锚帧。
从**模型生成锚帧**提取变化区域不属于读 GT，但其定位错误必须计入最终失败。

对应 W 也仅来自源视频。使用现有可验证的 Stream/DINO 特征匹配、source
optical flow 或两者组合；不假设历史失败的 Stream track head 已经恢复。
初始机制实验先用已知合成 warp 或确认存在的 flow/visibility GT，隔离匹配错误。
真实推理对应模块是另一个待验证依赖。VAE/DINO/RAE 传播器共享相同 W。
相机移动中的静态建筑及动态人物都按表面对应处理，不用运动幅度判别动静态。

生产适配先沿用9个RAE观测时刻0,2,…16，保持现有Stream上下文与518letterbox。
源17帧均可用于计算对应；奇数帧的效果通过源视频的双向可见对应补齐，无可靠
对应时保留该帧native提议，不能复制邻帧。native VAE仍按原来的17帧/4倍因果
时间压缩处理。机制对照先在9个共同观测时刻上测，再测完整17帧实际视频。

## 5. 可执行定义的联合投影

为避免自由生成新视频，优化变量只有源状态的小改动 delta_i。对一个源窗口，
同时求满足以下条件、且最接近当前编辑提议的状态：

    最小化：sum_i w_i ||R_i(delta_i) - b_i||² + beta ||Lambda_i delta_i||²
    约束：  锚帧上的 R_a 接近已给定锚帧的可见改动 a_a；
            可靠边上 V_ij(R_j - W_ij R_i) 的残差低于指定容差；
            若存在合法 source-region 条件，其外部改动受限；
            所有 delta 均处于固定尺度的信赖域。

b_i 为原生模型当前 clean estimate 相对 X 的编辑效果；未开始 native
采样的初始化，可以采用相同的 RGB 对应传播提议。各比较方法共享 b、锚帧、
W、区域条件。初始化、容差和最大迭代数在 dev 固定，不随测试 target 调整。

锚帧约束保留指定的编辑强度，防止 delta=0 成为“一致但不编辑”的解。
若输入本来就是 no-edit，目标效果为0，最小范数解就是delta=0。
约束对非线性 decoder 可能不可行，不能保证每次都能找到解。

实现首版：3个信赖域外层更新，每次最多4个矩阵自由内层迭代，按单帧 decoder
分块；只是起始预算，不是已测吞吐。置信不足的对应不施加硬一致性。新显露
且无可靠路径的区域保留原生提议并报告；不假造未观察到的3D表面。
若求解发散/不可行，回退原生去噪更新并记录该 case，不回退复制源视频。

真实遮挡/重现需可靠的跨时刻边；仅相邻 flow 链无法保证重现后身份一致。
correspondence 随机打乱/相邻-only/有无 visibility 必须作为消融。

显存实现先做块坐标更新：其他帧的已解码效果暂存，当前只对一个帧的delta
反传，随后重算相关图边；不同时保留9个518px decoder的反传图。图像编辑器、
Stream encoder、视频14B阶段串行暂存缓存，不同时常驻两套大型生成骨干。
普通source-RGB保持/合成规则必须给所有基线同样使用，不计作RAE贡献。

## 6. 写回真实去噪轨迹，而不是旁接辅助头

先从 calibrated R 得到原分辨率参考：

    Y_ref,i = clip((X_i + epsilon) * exp(R_i) - epsilon, 0, 1),

实际实现包含epsilon、范围裁剪与同一套有效区/letterbox反变换。此处源 RGB
是合法的编辑输入。这条输出不把全部画面重新走 RAE 重建；R=0 时精确返回 X。
它消除固定 AE 基础重建误差的直接叠加，不保证大改动没有 decoder 偏差。
加法版本 X + [D(z+delta)-D(z)] 也是必须比较的基线。

令 native flow 约定为 x_sigma=(1-sigma)u+sigma*noise，v=noise-u。
当前 clean estimate 为 u_hat=x_sigma-sigma*v，Y_hat=D_V(u_hat)。定义：

    Delta_u = E_V(Y_ref) - E_V(Y_hat)
    u_corr  = u_hat + alpha(sigma)*Delta_u
    v_corr  = v - alpha(sigma)*Delta_u/sigma.

native sampler 使用 v_corr 更新后续真实 latent。E_V 使用确定性编码；图像
范围/归一化/原分辨率/时间压缩约定严格按当前 Wan 实现。
初始只在sigma约0.3和0.15两个最近的实际scheduler位置启用，alpha在dev校准，
不在sigma=0除法。大噪声 clean estimate 不可信，不对纯噪声做 RAE 写回。

为何采用编码差值：直接覆盖 u_hat=E_V(Y_ref) 会混入 native VAE roundtrip
误差。此处若 Y_ref=Y_hat，则 Delta_u=0，去噪轨迹精确不变。这个恒等性
与 RAE 的 no-edit 源重建恒等性不同，二者不能混为一谈。
VAE 编码差值不保证 D_V(u_corr)精确等于Y_ref，必须测实际主输出的响应。

通过关闭写回、交换delta、交换锚帧、交换W，在相同seed下测最终输出变化，
检查 RAE 状态是否真正产生有方向的可见作用。不是仅检查梯度是否非零。
初版允许有推理优化开销；teacher 未取得收益前不承诺免开销最终模型。

## 7. 怎么训练、需要什么数据

先执行无新增大模型训练的算子实验，再决定是否摊销求解成本。

第一步：冻结已有 E_R/D_R，用合成可见对应和现有高质量 source/edit pairs
验证 H1 及算子可达性。并排展示 naive token warp、RGB effect warp、校准传输。
需要已有生产 decoder 和少量 source 编码；当前 bank 主要缓存 edited target，
所以 source RAE 编码是新缓存，不能直接把 target z 当 source z。

第二步：固定共享锚帧与 source-only W，在相同 VACE native backbone、scheduler、
步数、CFG、source条件下比较真实写回效果。另给等额推理计算的普通refine/
pixel-space projection基线，不能把多次14B调用带来的收益归给本算法。
纯文本端到端结果包含前端失败，不静默只保留成功锚帧；oracle-anchor 仅定位
传播上限，绝不进入主协议成绩。

第三步（只有前两步有效时）：用约6–8M参数的传输模块摊销delta求解。
候选为6层、宽256、8heads、FFN1024的source-conditioned时空/稀疏对应模块。
输入source C256、锚帧效果的粗传播、visibility、native提议及sigma，输出
标准化delta_i。它预测已经观察的源视频上的编辑动作，不从噪声预测未来视频。
目标为求解器状态和它实际写出的效果；delta并非唯一，不能只蒸馏其L2。
训练优先复现teacher的可见效果和最终native写回增量，不新增自由辅助readout。
该架构参数数目仍需实现时实算。

还可以把有效v_corr蒸馏进已有native LoRA，让最终推理不运行求解器；这是
第二种提速实现，先不同时建两套。蒸馏必须包含实际采样访问的状态，不能只
在clean GT附近学习后宣称整个采样过程有效。teacher有益是蒸馏的前提。

现有Senorita用于外观编辑监督，现有源视频用于对应与保持。无需所有训练
视频提供深度/位姿/分割。独立几何/运动评测需要一小批可靠对应/可见性或
人工确认案例。更强文本前端是可替换依赖，不是本项目已经具备的能力。

## 8. 最小实验矩阵与必须推翻自己的情况

机制实验固定锚帧、对应、分辨率、源数据和编辑强度，先不加VACE掩盖误差。

| 版本 | 唯一改变 | 回答的问题 |
|---|---|---|
| RGB effect warp | 直接传播同一编辑效果 | 是否只需普通像素传播 |
| naive C256/token warp | 不校准decoder响应 | H1是否真实存在 |
| 本方法，但复用anchor的响应 | 不使用目标源状态的局部响应 | 是否需要逐状态校准 |
| 本方法，native VAE | 同W/算子/锚帧，换codec | RAE是否必要，或是通用decoder方法 |
| 本方法，DINO-RAE | 同样的读写算法，换表示 | VGGT几何来源是否有额外作用 |
| 本方法，真实生产RAE | 目标帧校准并写回 | 主候选 |

VAE时域压缩与RAE逐时间状态不同，先做逐帧/匹配观测预算机制对照，再报告
完整时空codec实用对照；不能把时间分辨率、latent维数或重建质量差异全归为
VGGT几何优势。DINO-RAE同样要报告重建质量和表示大小，不能声称绝对公平只
因为输入图片尺寸一样。

端到端再比较：强共享前端+RGB传播+同VACE；上述baseline+直接VGGT/DINO监督；
Stage61式RAE辅助loss；我们的实际状态传输/写回。所有方法接受相同外部信息，
辅助特征baseline也获得同一对应提供的普通模板。写回和生成预算单独匹配。

评价采用两条主轴：编辑执行 vs 保持/运动，而非越接近source越好。画出
编辑强度—保持的Pareto曲线，保留无编辑负对照与所有失败例。H1度量在已知
对应的合成变换上计算；真实几何/动态用未参与构造W的评估器或可靠GT核查，
避免同一个teacher既提对应又给自身打分。

以下任何一种都足以否定相应主张：

- 校准只因缩小delta而更稳：不能支持方法有效；锚帧编辑强度/语义未达标。
- 不交换误差无明显存在，或简单线性/归一化已解决：H1动机不足。
- H1下降但真实native输出无收益：写回或误差选择失效，不继续扩大训练。
- RGB传播+同写回达到相同结果：RAE不是必要组件。
- VAE同算子一样好：不能声称RAE优越，最多研究通用decoder算子，仍需独立新意。
- DINO-RAE同样好但raw-feature监督差：支持可解码操作范式，不支持VGGT独有优势。
- 只在oracle锚帧或简单调色有效：尚未证明完整文本视频编辑方法成立。

## 9. 研究新意的准确边界

候选贡献是：测量并处理上下文表示中“编辑传输—解码不交换”的问题，构造
按目标状态校准、保持可见编辑强度、且真正写回视频去噪轨迹的操作。
这与继续减小一个teacher feature loss的优化对象不同，但还不是无人做过的证明。

已有构件不能算新：

- [TokenFlow](https://arxiv.org/abs/2307.10373)：跨帧特征传播用于编辑一致性。
- [LOCO Edit](https://arxiv.org/abs/2409.02374)、
  [Diffusion Pullback](https://arxiv.org/abs/2307.12868)：Jacobian/局部子空间用于图像编辑。
- [Trajectory Forcing](https://arxiv.org/html/2606.22527v1)：可解码RAE中间状态及交互编辑。
- [RFA](https://arxiv.org/html/2609.37330v1)：特征残差适配与冻结decoder训练。
- [V-RGBX](https://arxiv.org/abs/2512.11799)：intrinsic视频编辑与关键帧传播。
- [Geometry Forcing](https://geometryforcing.github.io/)、
  [GAD](https://arxiv.org/abs/2610.18779)：几何特征/对应监督。

所以“用Jacobian”“关键帧传播”“RAE可编辑”“加残差”均不单独作为创新。
代数、局部线性化与约束求解是工具；真正要成立的是本项目的机制证据与动态
编辑收益，并通过以上强对照排除更简单解释。未找到完全同名工作不是新颖性保证。

## 10. 当前交付边界

本次完成核心设计与原始文献核查，更新研究上下文。没有改动Stage61训练、
启动新集群实验、下载模型或宣称H1已在生产AE上成立。
一个标量非线性decoder的构造反例，以及native flow写回符号的数值校验已通过，
记录在D:/workspace/VggAE_DataAudit/vace_review_20261104/analytical_design_sanity.json；
它们只验证数学自洽，不是生产RAE结果、真实数据结论或算法效果证据。
下一步最有价值的是H1和读写可达性小实验；需要的额外输入只是source编码
和少量对应/锚帧，不需要先训练一个新DiT或重训整套AE。
