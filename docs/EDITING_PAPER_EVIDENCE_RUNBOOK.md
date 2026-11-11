# Stage69：三节点补齐论文证据

2026-11-11。当前预算 **3 节点 × 8 张 910B**，各阶段顺序运行。
固定 Stage67 自适应回溯、native RGB 条件不额外同步的版本。Stage68 同步路径
实际生效但未带来稳定质量增益，本轮不继续调它。尚未提交集群任务。

## 启动

保留现有 ModelArts 代码/权重准备方式，无须新的启动入口。三台节点执行同一命令：

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
PAPER_STAGE=smoke bash scripts/scale/69_complete_editing_paper.sh &&
PAPER_STAGE=all bash scripts/scale/69_complete_editing_paper.sh
```

smoke 使用旧 dev 的 3 例、6 步采样、2 次 SFT 更新，覆盖全部分支；不是质量门槛。
all 使用独立 namespace、完整训练预算和保留样本。新机器也可以重新执行同样命令，
按 receipt 恢复已完成阶段/案例。未完成的单例采样从同 seed 重启，不冒充恢复 UniPC 历史。
配置或代码改变后使用新 `PAPER_NAMESPACE`，避免混合不同方法的产物。

默认输出（另有 ModelArts OUTPUT_URL/scale 同步副本）：

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/editing_paper_evidence_v1/
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/editing_paper_evidence_v1_smoke/
```

必要输入仍是现有资产：Wan2.1-VACE-14B、StreamVGGT、Scene C256 AE、Stage60 完整
bank.json/12个 Senorita pilot 包、Stage64 prepared.json/source_scale、Stage65
训练缓存及 RGB/RAE selected.pt。均双读，先本地再两个持久化根；不重新下载全量数据。
LPIPS 使用已准备的 AlexNet 本地权重，缺少时明确记为 unavailable，不隐式联网。

## 训练和评估规模

| 内容 | 数据/预算 | 目的 |
|---|---|---|
| 新增无状态 SFT | 精确复用 Stage65 训练/monitor 清单、reference context、噪声采样、LoRA rank、学习率和最多1500更新的停止规则 | 排除“只是同任务 SFT 或参考更强”的解释 |
| 主表7组 | 最多128源 × 首seed | Native、Native+共享传播参考、匹配SFT、RGB无反馈、RAE无反馈、RGB完整、RAE完整 |
| 消融10组 | 输入预先固定的最多32源 × 首seed | 见下表，不根据生成胜负挑选 |
| 第二seed | 相同32源 × Native/RGB完整/RAE完整 | 采样稳定性；不重复计为新场景 |
| 自动评估/人评包 | 全覆盖；人评32源×3配对 | 目标相似度、独立保持/时序诊断和最终画面判断 |

没有新增 AE、DiT 或 RGB/RAE bridge 训练。SFT 没有 state bridge；**同训练预算不等于
同参数量，也不保证相同步数早停**，实际参数/更新/选择规则均记录。旧48例 SFT 不参与主表。
最多1312条最终视频，另128条共享前端输出；实际保留数不足则如实缩小并记录原因。
三节点用于全部阶段，不需要额外8节点。新权重加载、缓存、SFT和更多案例使本轮耗时
不同于复用旧输出的22分钟Stage67，不能据那个数字承诺总时长。

## 队列和独立恢复

1. CPU：pilot staging、源视频指纹、排除近重复、冻结清单。
2. 分别加载 T5、VACE前端、StreamVGGT/AE、VAE，准备源/参考/state/context。
3. 24卡仅训练一条匹配 SFT；阶段 ckpt、EMA、优化器和每 rank RNG 保存。
4. 每节点先串行拉取 selected 小权重，避免8个rank同时下载同一个ckpt；各臂新进程加载大模型。
5. 顺序完成主对比和消融；某臂失败继续独立臂，最后明确报告缺项，不伪装完成。
6. CPU/Gloo 自动评估；node0 CPU 汇总、两端媒体读回校验、人评包。

idle guard 与双写日志贯穿 CPU/等待阶段。所有相关阶段有字面 `DI_throughput`，
只计成功完成项；失败尝试单列。采样计时包含在线求解/解码/中间发布，加载/准备另计，
不把 guard 运算算进吞吐。内存使用沿用已运行的阶段隔离；不同时驻留 Stream encoder 和14B。
本地测试不等于已验证本轮910B/HCCL峰值显存。

部分恢复命令，仍在三节点同时执行：

```bash
# 仅补一条臂，其后统一重新评估和汇总
PAPER_STAGE=render PAPER_ARM=rae_full bash scripts/scale/69_complete_editing_paper.sh
PAPER_STAGE=evaluate bash scripts/scale/69_complete_editing_paper.sh
# 仅补汇总/媒体；不加载NPU模型
PAPER_STAGE=summarize bash scripts/scale/69_complete_editing_paper.sh
# smoke 的部分恢复：指定它的完整 namespace
PAPER_NAMESPACE=editing_paper_evidence_v1_smoke PAPER_STAGE=summarize bash scripts/scale/69_complete_editing_paper.sh
```

还支持 `PAPER_STAGE=inputs|sft|dry`。部分恢复从已有 manifest 读取 effective config。
脚本失败意味着缺产物/运行异常；不会用“低分”阻止其他训练或采样。

## 消融的精确定义

| 臂 | 只改什么 | 不能据此宣称什么 |
|---|---|---|
| rae_reinject | 在线优化0次，保留相同写回过程 | 不是彻底关闭反馈；用于区分重复注入与求解 |
| rae_no_temporal | joint_weight=0，保留图可达性/支持 | 不是移除全部对应信息 |
| rae_no_graph | 在线图为空且joint_weight=0 | 公共RGB参考仍有源flow；不是整个系统无对应 |
| rae_shuffle_graph | 打乱在线pull坐标，原参考固定 | 错对应压力测试，不是自然遮挡GT |
| rae_no_reliability | 在线观测/边confidence置1 | 缓存proposal和语义区域保持相同 |
| rae_bridge_only | 求解与VAE计算保留，只提交bridge修改 | 不把被丢弃的velocity修正记为实际写入 |
| rae_velocity_only | bridge state固定，只提交velocity | 首始state仍存在 |
| rae_zero_edit | 视频edit channels置0，source/支持/独立anchor prefix不变；无反馈 | 对照RAE base，避免在线反馈重新填回编辑 |
| rae_no_state | bridge bypass，已训LoRA/reference保留；无反馈 | 不等于从头训练无bridge的SFT |
| native_reference_60 | native+相同参考，60采样步 | 更多采样工作，不自动等于等墙钟/等FLOPs |

## 数据与评估边界

- 从旧 bank.manifest 的 reserved test 取每个source byte group一个prompt；旧train/dev明确排除。
- 冻结前做源视频首/中/末pHash近重复筛查，记录暴露源读取覆盖；均匀图不误判成近重复。
  没有parent-scene ID，**不能保证真正场景级不相交**；缺失筛查项透明记录。
- 新前端统一用 frozen native VACE 生成anchor，不依赖历史前端LoRA。与旧dev前端不同，
  新表内所有方法共享这个协议，不跨表拼百分比。无未来target当推理条件。
- PSNR/SSIM/LPIPS对数据集生成的paired target，只是参考相似度；target差分区域仅评估，
  不是语义GT。Farneback源flow不同于方法DIS，但也不是3D真值。静止/漏编辑可能骗过部分指标。
- 主表只用首seed；消融只用同32源同seed；第二seed单列。配对bootstrap匹配共同source/seed，
  先源内平均再按源重采样，失败/缺失覆盖率单独列。不可把帧或seed当独立样本。
- 输入运动分组固定为最大256px下源flow均值低于/不低于1px，不标成真实动静态或相机标注。

## 产物与待补结论

- `summary.json`：覆盖率、配对差值/95%区间、失败、实际成本、指标可用性。
- `tables/{main,ablations,robustness}.{csv,tex}`：实际测量表，LPIPS缺失保留空值。
- `report.html` / `comparisons/`：全部保留样本，而非只发布有利案例。
- `render/<arm>/seed42/<id>/`：lossless NPZ、三列视频、结果及每次feedback pt/视频。
- `media_sync.json`：输出目录和持久化镜像的每份媒体SHA/size读回验证。
- `human/index.html`：匿名A/B评审包；源/指令/随机顺序，各维度独立打分，浏览器导出JSON。
  只分发 `human/`，不发 `human_private/key.json`；建议5位独立评审者，人评尚未发生。

收集实际评分后，在本机用 `scripts/score_editing_paper_human.py --key <key.json>
--ratings <rater1.json> --ratings <rater2.json> --output <human_scores.json>` 汇总。
脚本拒绝重复rater/trial，按源平均与bootstrap；tie=0.5，neither单列且不充当获胜。

本轮能回答泛化、参考/SFT混淆、对应与写回信息作用、随机种子和成本问题。
**仍未补齐**VAE/DINO表示替换、token-only校准、公开外部编辑方法、独立3D评价和
重建各trick的因果消融。不能把这轮实现称作已经支撑SOTA或VGGT独有优势。
论文已安排对应字段，新分数必须在真实运行后填写。

## 本地验证

端到端显式小模型fixture覆盖实际VACE/UniPC、SFT反传/保存、全部17臂、目标文件删除后的
独立生成、统计seed配对、双读双写及损坏媒体恢复。真实Bash测试覆盖3节点参数、CPU隔离、
smoke独立配置、部分恢复及失败继续。合计55项相关测试通过（新套件8项、launcher4项、
已有求解器/反馈/媒体/VACE接口43项）；8个Python文件编译、Bash语法检查通过。
真实CLI逐phase分发也覆盖，避免仅直接调用函数测试掩盖启动错误。生产NPU/OBS仍待集群smoke。
