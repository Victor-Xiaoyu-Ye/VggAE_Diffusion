**项目审查与质量优先路线，2026-09-08**

本次审查基于本地仓库 `D:/workspace/VggAE_Diffusion`，审查前 HEAD 为 `c4459d252441d5ff177f7c0e55ff67e467a0ad9e`，分支 `ascend-910b`；实验来源是 `C:/Users/y50046448/Desktop/logs`。用户提供的旧路径 `D:/workspace/VggAE/_Diffusion` 在当前机器不存在。

用户本轮明确：两条路线都可以评估，优先生成质量；资源继续采用仓库中的 48 卡 Ascend 910B、ModelArts、OBS 和既有启动方式。这允许重新评估“最终必须替代 VAE”的研究约束，并不表示新模型已经选定或原生 I2V 已在 NPU 跑通。

**我的建议：将原生预训练 I2V + 几何监督作为优先验证路线，将现有 R7 保留为小预算研究分支。** 原因不是几何表示不可能生成视频，而是当前系统同时更换了视频表示、压缩结构、解码器、Wan 输入输出语言和训练目标，尚未建立一个可用的视频生成起点。继续同时优化这些部分，很难判断一次失败来自哪里。

本次没有重训模型、提交集群任务、改动训练实现、覆盖 checkpoint/cache 或放宽旧 gate。下文的架构和实验次序为建议，已测事实与待验证解释分开列出。

**原始实验到底说明了什么**

逐行解析了 31 个 `metrics.jsonl`，提取 313 条带 eval/gate 字段的记录，没有 JSON 解析错误。另读取 n1 的 `memory_status.json`、`denoising.jsonl`、`eval_samples.jsonl`，以及训练日志，并检查了 n1/旧 Wan 的保存图像。指标汇总不是对 checkpoint 的重新推理。

| 实验及记录位置（相对 logs） | 实测结果 | 能得出的结论 |
|---|---|---|
| `r7_t2_c192_v3/joint/metrics.jsonl`，12K | PSNR 24.213；LPIPS 0.1231；geo-motion cosine 0.8323 | 最后评估的重建已有结构，但不通过旧几何特征 gate。不能用历史约 23.9 的数字替代所有 checkpoint |
| `r7_t1_c192_probe_v1/joint/metrics.jsonl`，9.5K 最后评估 | PSNR 24.599；LPIPS 0.1082；geo-motion cosine 0.8667 | 取消时间折叠改善部分指标；没有证明生成已经可用 |
| `r7_t1_c192_geo112_tex80_probe_v1/joint/metrics.jsonl`，12K | 32 clips，PSNR 24.602；LPIPS 0.1083；geo-motion cosine 0.8922 | RGB gate 通过，旧几何特征 gate 未通过；没有证据支持继续仅扫 channel split |
| `r7_vggt_quick_geo112_tex80_v2/det_k1_n1/metrics.jsonl`，500 | generated-vs-AE PSNR 35.111；RAW 21.756；AE RAW 21.836 | 单 pair 的表示、确定性预测与解码组合能达到这个 pair 的 AE 水平 |
| 同目录 `det_k1_n16/metrics.jsonl`，1K | held-out 16 clips：RAW PSNR 19.510，copy 18.997；LPIPS 0.2344，copy 0.2598 | 确定性首帧预测略优于复制，证据有限；不是视频 diffusion 成功，也不是完整失败 |
| `r7_geo112_flow_n1_probe_v1/n1_plain_x0_n1_f1/memory_status.json`，500 | 四 seed 平均 RAW 12.823；vs-AE 12.893；copy RAW 17.041；normalized MSE 1.333；passed=false | 单 pair 自由采样尚未记住目标；生成问题远大于当前 AE 与原视频之间的差距 |
| `r7_wan13b_t2v_ctx1_fut4_v1/metrics.jsonl`，7K → 14K | EMA x0 MSE 0.6710 → 0.3577；生成 vs-AE PSNR 10.952 → 9.076；LPIPS 0.8153 → 0.8556 | 去噪指标变好，实际采样变差；不能继续只按 x0 loss 选 checkpoint |
| `r7_diffusion_t1_c192_ctx1_fut8_v1/metrics.jsonl`，6K | 生成 vs-AE PSNR 11.692，LPIPS 0.6965；expanded-geo cosine 0.0108 | factor-1 本身没有解决视频生成 |

不同实验的样本集、目标、解码上下文不同，表中数字不能直接排成统一排行榜。尤其 32-clip AE PSNR 24.60 与 n1 AE PSNR 21.84 不矛盾。旧 cache 只有首帧 RGB，因此旧 Wan 训练中的 RGB 指标是相对 AE_TARGET，而不是 RAW。

n1 图片的 AE_TARGET 能表达街道、建筑和行人，GENERATED 则是重复的彩色纹理。旧 Wan 手动样例也呈现强烈网格/彩噪。它们支持“生成链路尚未产生可用内容”，不能单靠图像反推出唯一数学原因。

**失败原因的证据等级**

1. **已确认：重建质量不足不是当前灾难性生成的充分解释。** 重建影响最终细节，但不能解释同一 pair 的确定性模型达到 35.11 dB vs-AE，而 flow 只有 12.89。注意 AE 重建是 codec 基准，不是任意感知指标的严格数学上界；更不能要求合理的随机未来逐像素等同唯一 GT。

2. **已确认：去噪/自由采样之间存在大差距；具体原因尚未定位。** n1 在数据时间 t=.9 的 x0 MSE=0.04093，实际同噪声 `x_t/t` 基线为 0.01249；t=.99 时模型为 0.03602，基线为 0.0001032。接近干净端仍引入显著误差。代码使用 `x_t=(1-t)ε+t z`，plain-x0 速度为 `(z_hat-x_t)/(1-t)`；x0 误差会进入速度，但仅凭分母不能断言积分必然爆炸，因为最后一步 Euler 的步长也会缩小。现有 v2 的预条件化/direct-velocity 分支值得测试，不应预设哪个必胜。

3. **已确认：确定性成功与新 flow 不是单变量对照。** `models/single_target_generator.py:69` 显式输出 `anchor + residual`，直接从 anchor tokens 预测；`models/r7_flow_probe.py:98` 起使用独立 anchor/noisy 投影、拼接 tokens、零初始化直接输出头，且标准化、模型宽度和时间条件也改变。现有证据不能把差异全部归因于 Gaussian noise 或 latent 流形。条件通路是否被有效使用、位置表示是否支持噪声端预测，都值得检查。

4. **已确认：当前 Wan 路径不是原生 Wan I2V。** `models/wan_compact_adapter.py:153,182,284` 用新 input/output projection，绕过 `wan.patch_embedding` 与 `wan.head`。网格、时间语义、目标参数化也与预训练不同。最新代码已经加入 `reverse_flow_time=True`、anchor memory 等修正，但桌面旧 Wan 结果不能当成这些新配置的评测。适配后仍保留多少预训练能力，只能通过原生基线及受控适配实验回答。

5. **已确认：所谓 geometry-motion 是代理指标。** `train_causal_dual_tokenizer.py:182,548` 比较 `geo_rec` 与 `geo` 的时间差分余弦；其中 geo 已经过 CompactCompressor。它没有运行相机/深度/点图预测头，也没有计算重投影或真实三维运动。0.95 可以继续作为旧实验的特征保真约束，但不等于物理几何正确性，0.892 也不能单独判定表示不可用于视频生成。新路线不应把跨过这个数字当成所有进度的前置条件。

6. **代码结构事实：geo112|tex80 不是严格可分离的最终 latent。** `models/causal_dual_tokenizer.py:66` 将两流拼接后送入全通道时空 codec；`models/causal_temporal_codec.py:65,68` 的普通 Conv3d 混合通道。直到 temporal decode 后才按 geo/tex 分割。因而生成 latent 的前112维并不自动代表纯几何，后80维也不自动代表纯纹理。当前联合生成仍然合法，但若下一步要独立预测几何/纹理、做几何专用噪声或前112维 ablation，必须先验证可辨识性，不能只凭命名。

7. **代码结构事实：完整 RGB decoder 并不严格因果。** `models/dual_stream_decoder.py` 的 TemporalAttnBlock 没有 causal mask。当前固定窗口联合生成全部未来帧可以使用这种解码器；单帧 probe 的重复 suffix 也是明确且无真实未来泄漏的诊断契约。它不等价于在线流式解码器，不能直接推导长视频 rollout 正确。本次 pair 的完整 AE 与重复 suffix AE 仅差约0.218 dB，不能用这一点解释约9 dB的生成损失。

8. **已排除一个当前怀疑：这次 n1 不是明显未加载 encoder。** `logs/train_node0.log` 显示需要的 1210/1210 encoder keys 匹配。它不反向证明所有历史 checkpoint 都正确加载。

EMA 仍是旧 Wan 的混杂因素；不能用默认 .9999 推断所有运行实际使用值。仓库已有回放工具，需同一 checkpoint 的 online/EMA 比较后再归因。此前文档的“均值 latent 复现高 motion cosine”分析本次未重新加载 tensor 复算，不能把它算作新增实验结果。

**推荐主线：保留视频生成模型的原生接口，让 VGGT 提供几何约束**

```text
真实首帧 + caption
    → 原生 I2V 条件路径
    → 原生视频 DiT / noise schedule / latent space
    → 原生视频 VAE decoder
    → RGB 视频

训练期：真实训练 clip → 冻结 VGGT/StreamVGGT → 几何 teacher 特征
                                      ↓
                      与 DiT 中间表征做投影对齐
                      梯度更新 LoRA/adapter
```

第一版只增加训练期几何监督。保留 VAE、patch embedding、原生输出头、时间语义、文本编码和首帧条件，避免再次承担恢复整个视频 prior 的成本。全 clip 的几何特征是监督标签；推理输入只含首帧/caption，不输入真实未来的几何信息。

这条范式有直接先例：Geometry Forcing 通过方向和尺度两种对齐目标，把 VGGT 表征引入视频 diffusion 的训练。[作者项目页](https://geometryforcing.github.io/) 因而“加一个 VGGT loss”本身不是新的论文贡献；对于本项目，它首先是强基线。进一步贡献要落到动态区域、遮挡、几何置信度或实测的几何/画质收益上。

模型优先选**原生支持 I2V**且能在现有 NPU 环境跑通的 checkpoint。现有 Wan2.1-I2V-14B 若权重已可用，可先做原生推理基线；需更小训练基座时可评估 Wan2.2-TI2V-5B。官方仓库提供这些原生 I2V 路径，5B 是独立的原生 VAE/模型组合，不应把它接入现有 R7 WanCompactAdapter 后称为原生基线。[Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)、[Wan2.2 官方仓库](https://github.com/Wan-Video/Wan2.2)

这里没有声称 5B 在所有任务优于14B，也没有把官方 GPU 显存数当作 910B 保证。第一选择由“权重可用性、910B smoke、生成样例、实测显存”共同决定。已有 T2V-1.3B 可以保留为低成本控制，但它在你们的自定义接口下不是原生 I2V 的替代基准。

实现时先选少量 DiT 中间层，加入轻量投影头，训练 LoRA + 对齐头。以原生 diffusion loss 为主，增加小权重 angular/scale alignment，并记录两类梯度量级。只训练一个与生成路径断开的读出头不会改善视频，几何损失必须能回传到影响生成的 LoRA/adapter。

teacher 特征须按真实时间区间和空间坐标对齐到原生 VAE/patch 网格；记录 resize/crop 变换。不要把9帧1秒的 R7 时序索引直接当作原生模型的帧率与 temporal patch 语义。第一步保留原生支持的帧数/分辨率，再按物理时间评测共同片段；短窗训练的时长和帧率需要另作受控适配。

训练初期使用可靠 teacher 特征和置信度掩码；未知/强动态区域不能硬套静态世界的 rigid reprojection loss。如果以后加入几何输入 adapter，先只用真实首帧能获得的几何；未来相机轨迹需用户给定或由模型预测，不能从评估 GT 偷取。

原生 baseline 跑通后依次比较：

| Arm | 修改 | 回答的问题 |
|---|---|---|
| B0 | 原生 I2V，不训练 | 基座本来能生成什么 |
| B1 | 同一基座，SpatialVID 小规模 LoRA，仅原生 diffusion loss | 数据域适配是否保持/提升质量 |
| B2 | B1 + 训练期 VGGT 对齐 | 几何监督是否有净收益 |
| B3（B2有效后） | B2 + 首帧几何条件 adapter | 显式几何输入是否提供额外收益 |

B1/B2 的数据、分辨率、时长、优化器步数、LoRA 容量、seed 和采样配置一致。B0 是质量参照，不是假装算力匹配的训练对照。若有论文目标，再加入相同预算的语义特征教师（如 DINO）控制，证明收益来自几何，而不仅仅是增加表征监督。

Gen3R 还提供几何与外观 latent 对齐后联合生成的研究先例；它处理场景级3D生成，不能直接视为动态 I2V 已解决。[Gen3R 论文](https://arxiv.org/abs/2601.04090) 在 B2 之前不建议立即加第二个需要从噪声生成的几何 latent 流，否则又把几何预测能力变成 RGB 质量的前置瓶颈。

**R7 分支：先用可解析对照结束“到底是不是 sampler”的争论**

保留当前已完成 geo112|tex80 checkpoint，不改 cache 统计、不追加 AE 长跑。已有 stage26 的预条件化/direct-velocity n1 是合理的后续，但前面应补两项很便宜的验证：

1. 在实际 accelerator dtype、normalization、sample_flow 和 decode_full 链路中，使用输出固定真实目标 z 的 oracle。四个 seed、1/30/60步都应到同一个目标；这是 sampler 管线检查，不是生成成功。
2. 将已训练的确定性模型 `g(anchor)` 当成与 t、x_t 无关的 x0 predictor，接入同一个 sample_flow。先把原始 anchor 反标准化给旧模型，再把输出按新 target stats 标准化。对于 Euler 网格终点为1，最后一步满足 `x_next = x + (1-t)*(g-x)/(1-t) = g`，理论输出应与直接运行 g 一致。若不一致，优先查归一化、dtype、模型装配和解码契约。

第二项既不读取真实未来，也不训练新 diffusion，但只是确定性预测经 flow 的等价实现，没有学到随机未来分布。它可利用现有35.11 dB的 n1基准，明确证明“从 Gaussian 开始经过这套接口”本身是否能保留已知好输出。

之后才比较已实现的预条件化/direct-velocity random-noise n1；同一训练集、step budget、online/EMA、四seed，同时看 t=0/.01/.1/.5/.9/.99 和沿采样轨迹的误差。oracle-start、fixed-path seen-noise、unseen-noise 需要分开，不能互相替代质量 gate。

若条件通路弱，可增加一个独立 arm：`z_hat = g(anchor) + correction(x_t,t,anchor)`，correction 零初始化，从已知可用预测起步；训练的主目标仍是绝对标准化 target。这是模型输出的 skip/preconditioning，不是重新采用历史 `z_future-z0` residual-cache 契约。先冻结 g，验证 n1/16，再考虑是否需要联合训练。

单样本阶段不要求多样性，所有 seed 收敛到同一 pair 正确。多样性和泛化在更多训练样本及未参与调参的条件上测试。原有 generated-vs-AE>=30 dB 的 n1记忆 gate 可以保留；不能把它直接用作无确定未来约束的真实视频生成 gate。

GLD 证明几何基础模型特征可以用于多视角生成，VGGT-World 则报告 clean-target 参数化对其几何预测有帮助；二者都不能直接推出“当前强压缩双流 R7 是最优动态 RGB video latent”，也不能据后一篇禁止在本项目测试 velocity。[GLD](https://arxiv.org/abs/2603.22275)、[VGGT-World](https://arxiv.org/abs/2603.12655)

**评测要改成能够支撑项目目标的形式**

保留旧 gate 作为历史契约；新路线使用独立评测，不写假的 `gate_passed.json`。

- codec/确定性记忆：RAW 与 AE_TARGET 分开，PSNR/LPIPS 有效；报告完整序列与可部署前缀的解码差异。
- 随机 I2V：主体/背景稳定、运动合理性、文本遵循、静止率、彩噪/融化率，以及多个 seed 的条件内差异。PSNR/LPIPS 相对唯一 GT 是辅助配对指标，不能把它们作为唯一主分数。
- 几何：在可信静态背景测相机/深度重投影和跨帧一致性；动态前景用可见性/遮挡感知 tracks 与运动指标。报告有效匹配比例和失败率，避免“只有少数容易点被评分”造成虚高。
- 用独立几何估计器作交叉检查，并保留人工盲评。仅用训练 teacher 再评分会有自我偏好；低重投影误差也可能来自静止/复制，须与运动量和条件遵循联合解释。
- 固定 first frame、caption、物理时间和空间变换；native resolution 质量与统一尺寸的公平比较分开。
- 既有64个反复查看的 eval clips 作为开发集；最终另留按 source-video 隔离、未用于选模型/门槛的测试集。大规模 FVD 等指标在足够样本数后才使用，不对16/64个短片给过度确定的结论。

推荐从16个分层开发片段、每个2个seed做便宜筛查（静态场景相机运动、动态主体、遮挡/显露分别覆盖）；候选再做64 clips×4 seeds。相邻 checkpoint 的比较用同seed；几何和画质都报告配对差值/置信区间，不预设脱离 baseline 的“必达分数”。

**48×910B 的执行顺序**

沿用 `scripts/spatialvid_config.sh`、`scripts/lib/modelarts.sh`、`scripts/lib/spatialvid.sh` 的路径发现、MoXing staging、HCCL topology 和输出持久化。`/cache` 是临时空间；输出仍写 OUTPUT_URL 并 mirror，cache 单独版本化。新主线需要独立 wrapper，不能直接让现有 stage16 吃 native VAE latents，因为它明确要求 R7 manifest/stats/decoder gate。

| 顺序 | 工作与建议规模 | 停止/晋升依据 |
|---|---|---|
| 0 | R7 oracle 与 g(anchor)-flow 对照，1 NPU；不重新编码全量数据 | actual sampler/decoder 与直接目标一致，否则先修契约 |
| 1 | 原生 I2V 1 NPU 推理 smoke，必要时验证节点内分片；16 clips×2 seeds | 能生成完整有限值视频、首帧条件正确、实测显存与速度可接受 |
| 2 | 原生 LoRA 单卡前后向 → 8卡 save/resume → 48卡短 smoke | HCCL、梯度更新、混精度、恢复、OBS 和 PNG/MP4 都通过 |
| 3 | B1/B2 使用同一10K子集，先给每 arm 约1–2K optimizer steps的诊断预算 | 每隔数百步做便宜样例检查；候选做64×4，无收益/持续退化不盲目延长 |
| 4 | 候选跨新样本复现，才扩到完整数据 | 画质保住且几何收益可复现，排除静止和条件泄漏 |

这里的step预算是提议，不是已验证收敛时长；先根据 smoke 测得的 step time、吞吐、峰值内存确定卡时。48卡通常是6节点×8卡，DDP不会把单卡显存合成一张大卡。14B全参数复制训练不应重演；先冻结基座、LoRA、activation checkpointing、离线text/teacher，再根据实测决定是否需要NPU可用的分片方案。不能仅因GPU实现支持FSDP就宣称NPU已经支持。

如平台只能整批分配48卡，小样本步骤按既有stage25/26只让node0/device0执行，其余退出；原生基线推理在单卡可容纳后，可分配独立样本给各卡，不必建立训练DDP。

**本次验证范围与下一次最有价值的交付**

已完成原始指标汇总、关键代码/文档审查、样例目视检查，以及仓库 `scripts/test_r7_flow_probe.py` 的公式/静态/CLI部分：PASS。本机Python没有torch，脚本明确SKIP模型、梯度、实际sampler和normalization tensor测试。没有重新加载/运行历史权重，没有新增910B质量结论。

下一次应交付两个小而可检验的结果：实际R7管线的 oracle/g(anchor) 回放，以及原生I2V在相同开发场景上的视频样例。前者决定R7下一步应该修哪里，后者给几何增强路线一个真实质量起点。两项都比再次扫AE通道数或续跑旧Wan更能缩小决策的不确定性。
