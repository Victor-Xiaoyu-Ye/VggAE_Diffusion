# Stage58：上下文／尺度诊断与未来几何条件的小规模对照

2026-10-28。当前阶段固定 4 帧观察、5 帧生成，三个节点各 8 张 Ascend 910B。
这不是完整重训或可部署的新模型：先确认表示的一致性，再测未来 geo 特征是否能
帮助纹理生成。所有 oracle 输出明确标记，不作为普通视频生成结果。

## 启动

三个节点使用同一提交与原有 ModelArts 路径配置，启动同一条命令：

```bash
bash scripts/scale/58_train_scene_geo_texture.sh
```

默认 `all` 按顺序完成：独立几何诊断 → 共享小数据缓存 → baseline 两步预检和
新进程恢复训练 → oracle_geo 两步预检和新进程恢复训练。无需另行下载模型、
标注或原始视频；审计使用已保存的 raw RGB 与原 latent，不用有损预览。

可选操作：

```bash
# 只打印路径和运行计划
SCENE_GEO_STAGE=dry bash scripts/scale/58_train_scene_geo_texture.sh

# 独立 smoke namespace，24 个训练窗口、4 个评估案例、3 次更新
SCENE_GEO_STAGE=smoke bash scripts/scale/58_train_scene_geo_texture.sh

# 只做上下文／尺度诊断，不训练
SCENE_GEO_STAGE=audit bash scripts/scale/58_train_scene_geo_texture.sh

# 相同配置/namespace 恢复训练，跳过审计和两步预检；共享数据仍会检查恢复
SCENE_GEO_STAGE=train bash scripts/scale/58_train_scene_geo_texture.sh
```

`all` 已含真实两步保存、新进程恢复，无需为了启动正式实验额外提交 smoke。
只做 `audit` 后可用 `train` 接着跑。改代码、数据、配置或源权重须换
`SCENE_GEO_NAMESPACE`；相同任务中断后复用原 namespace，避免重复训练。
不要将 `_smoke` 数据目录用作正式数据。默认配置：
`configs/scene_geo_texture_pilot_v1.json`。

## 两组究竟比较什么

| 设置 | baseline | oracle_geo |
|---|---|---|
| 源权重 | Stage53 online step31069 | 同左 |
| AE／归一化 | production C256、原始统计，全部冻结 | 同左 |
| 观察条件 | 原始前四帧 latent | 同左 |
| 额外条件 | 全零、presence=0 | 未来五帧 GT 编码得到的 geo128、presence=1 |
| 模型与参数量 | 831,421,056 | 相同 |
| 训练目标 | 原始 C256 联合 x0 目标 | 相同 |
| 解码输入 | 全部由模型预测的未来 C256 | 同左，绝不复制 GT geo 到解码结果 |
| 训练预算 | 2000 次更新 | 相同 |

新增 adapter 只有 115,712 参数，约为原 831,305,344 参数的 0.014%。两组同初始化、
同样本顺序、同训练噪声与更新数；baseline 也保留相同 adapter，其输出始终由
presence 门控为零。oracle adapter 末层零初始化，因此两组源模型输出一致。
原主干 peak LR=5e-6，adapter=5e-5，warmup100，accum4，有效 batch=24×4=96。
保持原梯度检查点、FP32 EMA、AdamW 和 FP32 Euler 状态，避免同时改变多个因素。

源 latent 来源是同一窗口的因果编码，不是把未来五帧另起序列编码。每一帧有
共同首帧参考及过去上下文。这个约定不保证物理几何准确，也不代表 latent
标准化等于几何配准。未来 geo 可能携带外观：有效也只能先证明未来特征有帮助。

模型前向峰值显存尚未在本次 NPU 实测。主干、序列长度和原生产配置一致，额外
adapter 状态很小；诊断模型与训练模型在不同进程运行，不同时占卡。不要用之前
约 28.88 GiB 的单节点诊断累计峰值冒充训练显存。两步预检保存真实模型统计和
训练峰值，随后新进程恢复。

## 数据与阶段预算

- 从 Stage53 原街景训练缓存按确定性 shard 顺序读取最多 2048 个合法候选窗口，
  不是全量 HQ。重复 ID+frame_ids 去重；所有评估 ID 排除；格式坏样本逐条记录。
- 选 512 个窗口：一半取较大的归一化 texture 相邻帧变化，一半从剩余候选按固定
  hash 抽取。这是相机运动、物体运动及亮度变化混合的代理，不是语义动态 mask。
- 候选来自有边界的 shard 池，不宣称整个 32768-window 训练集的均匀抽样。
  `prepare/selection.json` 保存真实来源、frame_ids、分层和得分，便于复核。
- 一个 CPU leader 准备两组共享的 compact cache，每 shard 最多 64 窗口。其他
  节点由协调器等待，没有 HCCL 集体操作等待 CPU 扫描；所有卡均有 idle guard。
- 候选不足或部分源数据损坏时，实际数量明确记入 manifest，非空即可继续。
  输出写入失败不能当作坏输入悄悄跳过；原 source 永不修改。远端 compact 成功
  后只删除本任务私有的原 shard 临时副本，避免一直保留整库。
- 全部原始 64 个评估案例保留，包括 case041，绝不因画质差删掉。监测前 8 例，
  最终 64 例×2 seeds×online/EMA；GT 分支异常与正常案例须分层解释。

2000 步是同曝光的诊断预算，不是宣称已经收敛。每 500 步保存 full optimizer/
EMA/RNG checkpoint、轻权重和监测视频，结束再保存 final。不存在视觉阈值中止
后续独立实验。此前相近训练设置 5000 步约 7 小时，只能据此粗估两组训练更新
合计约 5.6 小时；本次缓存准备、评估、同步和实际 NPU 速度另计，不承诺固定时长。

## 上下文／尺度审计

8 个原始案例 0/4/7/11/24/36/41/48，每例三个任务，共 24 个独立任务，按任务序号
分给三个节点，每节点仅 NPU0 执行诊断。其余卡等待阶段也受占卡守护覆盖。

1. **表示与重编码**：full9、独立 prefix4、保持 prefix 不变但反转/翻转未来图像、
   单帧 anchor。比较实际生产四层 token 与压缩 geo/texture；保存原缓存与重新编码
   差异，以及重新 BF16 存储的量化差异。raw 来自 uint8 缓存，原编码前 float 输入
   不在缓存中；不能将这种微小差异直接叫做数据损坏。
2. **真实序列几何**：现有完整 StreamVGGT 的 camera/depth，加独立 LK 对应。
   固定观察前缀的首相机和深度中位数作为共同参考，检查 full/prefix 输出与相机
   重投影残差。真实 LK 点可能在运动物体上，残差不能一律当静态尺度误差。
3. **重复首帧**：相同画面是零图像运动对照。拟合整个后续深度体积的一个全局 scale，
   分别报告拟合前与剩余误差。拟合仅解释误差，不修改相机、latent 或训练目标。

所有模型仍为原已部署权重；生产 encoder 和完整 teacher 都记录大小及相同方式的
采样 SHA，与 Stage57 已知权重核对。每个 forward 后保存中间结果，解码或视频导出
失败时仍保留之前的数值证据。没有新 learned tracker 或阈值降低。

现有 Stage57 payload 的 CPU 重算已发现：重复图最佳单 scale 后，相对深度 RMS
仍约 case000 10.77%、case004 9.26%。说明该样本的变化不是纯全局尺度偏移。
这是现有输出的重算，不是新的 NPU 前向或物理真值精度评测。

## 如何判读训练结果

除了 RGB/AE/COPY 预览，训练与评估分别报告 geo 与 texture MSE。**oracle 的 geo
误差下降本身不算目标达成**，因为它收到相应条件；必须看 texture 误差、动态轮廓
与真实生成画面。原无条件/前缀任务仍有多解未来，L1 不能代替视觉质量与多样性。

最终 oracle 对 8 个指定案例、首个 seed、online/EMA 保存同噪声的 correct/zero/
shuffle 条件对照，shuffle 使用另一个案例同 codec 的未来 geo。每项都保留视频、
生成 latent、变化与误差指标。没有 donor 时显式报告缺项，不伪造错误条件。

- correct 比 baseline 和 wrong 明显改善人物，且 texture 误差同步改善：未来特征
  条件确有价值；仍需用推理可得的轨迹/预测几何替代，并排除外观泄漏后才谈方法。
- 仅 geo MSE 下降、纹理和画面没改善：不能称为成功，提示当前条件入口或表示组织
  没有效引导外观；不是继续扩大同一训练的充分理由。
- 训练子集有收益、held-out 无收益：局部拟合不等于泛化，应检查条件组织和数据覆盖。
- cached AE 与重编码 AE 对 case041 差异明显：继续定位缓存/编码链路；两者均提亮
  则指向 codec 重建问题。此审计不替换原 targets，也不重新训练 AE。

## 输出、恢复与 DI

默认位于原 local/primary/mirror 的 `scale/` 下：

```text
scene_geo_texture_pilot_v1/                 # 三节点 launcher 日志/阶段码/idle guard
scene_geo_texture_pilot_v1_audit/           # 24 个独立诊断、summary、数值/图/视频
scene_geo_texture_pilot_v1_data/            # 一份共享数据/选样清单/坏输入记录
scene_geo_texture_pilot_v1_baseline/        # 无未来特征的配对训练
scene_geo_texture_pilot_v1_oracle_geo/      # 明确标注的未来特征 oracle
```

训练结果主要看 `eval/step0000500/` 至 `eval/step0002000/`，最终 summary 和
各 `caseNNN_seed*.mp4`；oracle 还有 `_GEO_ZERO`、`_GEO_SHUFFLE`。第四列仍是 COPY，
本来就静止，生成是第三列。checkpoint 在每组 `train/`，含 model family、条件
分支、原归一化签名、优化器、EMA、游标和各 rank RNG。

所有阶段保留双读双写。恢复时检验 payload 并修补缺失镜像；本地历史训练快照仅
在双副本成功后按原策略保留近期版本，远端阶段快照保留。输入坏 shard 支持跳过
并写日志；若运行期间两组的 bad-shard 状态不同，比较时须披露，不能再声称完全
相同曝光。不要手动改已运行配置来追加步数；新阶段须明确 source 和 namespace。

诊断组件失败继续其他组件；整个审计失败也不阻断独立训练。某训练 arm 运行失败
继续另一 arm；缺少非空共享数据才跳过两组训练。各阶段协调器统一返回状态，
最终非零码不会被后续成功覆盖；退出码 0 不等于好画面。

DI：准备报告缓存 latent 帧/秒，训练报告每 NPU 未来 token/秒，评估含解码和
额外消融工作，分开注明单位。诊断最终 action 计时不包含最终 seal/report 发布，
中间 phase 计时也不包含模型加载；报告中明确写出范围，不当作纯模型速度。
占卡运算不计入 DI。默认空闲 600 秒触发 guard，避免连续空置两小时。

## 本地验证范围

2026-10-28 共 74 项相关测试通过：72 项集成回归，随后对更新后的 15 项几何测试
与新增 1 项真实两进程重启测试再次验证（重叠测试只计一次）。13 个相关 Python
文件编译通过，新 Bash 启动脚本语法与实际 stub 执行通过。日志保存在
`D:/workspace/VggAE_DataAudit/scene_e2e_20261028/stage58_tests.log` 与
`stage58_geometry_ddp_tests.log`。

CPU 合成几何、真实 tiny C256 模型优化/采样、future texture/RGB 隔离、strict warm
start、分支 metadata、两步保存后进程重启、实际两 rank Gloo 模型与本地 EMA 一致、
中间视频失败修复、双镜像恢复、共享数据与坏样本处理、实际 Bash launcher 及
torchrun 参数边界均有测试。编译和 Bash 语法检查也包含新入口。

这些不是生产 NPU、真实 OBS 或视频质量通过的证明；不能提前宣布 scale 已校准、
oracle 有用或论文方法已经成立。

## 2026-10-28 部分结果复核

只读检查用户下载到 `D:/中转站/scale` 的快照，未访问 OBS，也不能据此确认集群此刻
仍在运行。审计 24/24 完成，准备结束，baseline 启动日志到 1300/1310 步；保存的
评估及 checkpoint receipt 到 1000 步。还没有 oracle_geo 目录或全流程完成记录。
因此下面是前缀基线续训的结果，不能用来评价未来几何条件是否有效。

### 配对生成质量

仅比较监测集前 8 例、seed101；已经逐例确认 frame_ids、noise_seed 相同。
所有误差取未来五帧，采用原评估实现。不要与旧 64 例或 50 例均值直接比较。

| 权重 | RGB L1↓ | geo MSE↓ | texture MSE↓ |
|---|---:|---:|---:|
| 原 Stage53 online31069 | 0.101290 | 0.050169 | 0.378293 |
| baseline 500 online | 0.098120 | 0.046359 | 0.357185 |
| baseline 1000 online | 0.097128 | 0.045707 | 0.350060 |
| baseline 1000 EMA | 0.097332 | 0.045731 | 0.351024 |

相对源权重，1000 online RGB L1 下降 4.11%、texture MSE 下降 7.46%；相对 AE
L1=0.050003 的差距仅闭合约 8.1%。6/8 例 RGB L1 改善，但横穿车辆 case001 从
0.130864 变为 0.137507，行人 case004 从 0.100233 变为 0.100652。已查看全部八例
第4/6/8帧的同位置对照，车辆形变、人物腿部重影仍然明显。online/EMA 相近。
生成未来帧间变化代理从 0.061977 降到 0.059365，GT 为 0.082556；锐度代理基本
未变，不能据 L1 改善宣称动态恢复。上述代理均不等于物体运动或感知质量真值。

### 上下文、缓存与尺度

- 八例修改未来 RGB 后，观察前缀四层 VGGT token 以及 geo/texture latent 均完全
  不变。full9 与独立 prefix4 的 geo 原值相对 RMS 为 0.354%–0.412%，texture 为0。
  当前证据不支持“大幅前缀上下文错位”解释主要生成失败；不是物理几何准确性证明。
- 缓存与重新编码的 geo 标准化 RMS 为 0.005426–0.006439，texture 为
  0–0.00004121；八例重建 L1 的最大变化仅 0.00005195。
- case041 发白在重编码后复现：cached/reencoded L1=0.309495/0.309547，
  PSNR=9.519/9.518dB；其余七例平均 PSNR 约24.861dB。这里为全部九帧、未裁剪的
  decoder 输出，不能与旧 future5/clipped 指标混比。旧缓存错配不足以解释异常；
  AE权重、预处理及数值实现尚未进一步区分，也不能由一例宣布整个AE不可用。
- 重复首帧的 LK 位移严格为0，拟合一个序列 scale 后深度相对 RMS 仍为
  5.38%–10.77%，最后帧刚性重投影中位残差3.23–9.54px。这是教师自一致性误差，
  不是相对几何真值精度；不能只做一个 scale 校正便视为可靠运动监督。
- 现有几何残差分类在八段真实视频只给7条 dynamic tracks，却在重复静止图上
  给6条假 dynamic tracks；真实视频4608条候选中3765条unknown。该结果不足以
  建立可靠动态mask，unknown不能当静态。本次未把这些mask加进训练。

### 运行状态及证据边界

512窗口来自498视频、2048候选，256高latent变化+256随机剩余，64eval保留，
准备日志无坏输入。扫描137个源shard，实际compact文件135个；文件数不是窗口数。
可解析的3010条训练记录均finite，没有梯度裁剪事件；baseline额外条件及注入
严格为0，符合设计。记录峰值为节点0约25.40GiB、其他节点约22.03GiB，最新DI约
1393–1395未来token/s/NPU，连续更新约3.9秒/步，评估及同步另占时间。
双写状态均报告成功；24卡守护正在监测，快照未显示两小时空闲回收或退出。

736个本地实际存在的sealed文件通过大小及SHA校验，192个MP4全部正常解码
（184个九帧，8个预期四帧prefix视频），24个PNG正常。本地没有实际PT文件，
430个PT只有receipt；不能据此独立验证checkpoint、latent张量或完整恢复。
审计张量结论来自校验通过的运行报告，未在本地重新计算大型payload。

建议保持当前配置，完成既定baseline2000步以及oracle_geo2000步和正确/置零/
打乱条件对照，不在中途换AE、数据或配置，也不因这4.1%的改善自动延长同一基线。
下一判断必须看oracle是否真正改善texture和动态画面，并且优于错误条件。

分析与八例对照图位于 `D:/workspace/VggAE_DataAudit/stage58_partial_20261028/`：
`paired_metrics.json`、`visuals/caseNNN.jpg`、`audit/audit_findings.json`、
`training/review_metrics.json`、`integrity/summary.json`。本次只更新文档，训练代码
和用户结果文件均未修改。

## 2026-10-29 完整结果复核

用户更新的 `D:/中转站/scale` 已包含两组完整结果。baseline与oracle_geo均到
2000步，停止原因 `step_cap`，全部三节点及各阶段exit0；这表示按预算完成，
不表示已收敛或达到质量要求。未访问OBS，未启动新训练，训练代码未改。

### 同案例、同噪声的最终对照

下表每行64个独立视频案例×2seed，frame_ids、noise_seed和采样配置逐项匹配。
两组源模型评估完全一致。这里是完整64例，不能与上节前8例均值直接混比。

| 权重 | RGB L1↓ | geo MSE↓ | texture MSE↓ |
|---|---:|---:|---:|
| source31069 online | 0.0894004 | 0.0436140 | 0.3573332 |
| baseline2000 online | 0.0856243 | 0.0408064 | 0.3268305 |
| oracle2000 online | 0.0802864 | 0.0220528 | 0.2730680 |
| baseline2000 EMA | 0.0856323 | 0.0407870 | 0.3275851 |
| oracle2000 EMA | 0.0805720 | 0.0223859 | 0.2753965 |

AE参考L1为0.0438093，COPY为0.1049647。oracle相对baseline online改善RGB
6.23%、texture16.45%、geo45.96%；EMA对应RGB5.91%、texture15.93%。115/128次
输出RGB改善、125/128次texture改善；先对两seed平均，则58/64案例RGB改善、
62/64案例texture改善。按视频案例bootstrap10000次，RGB相对改善95%CI为
[4.59%,8.03%]，texture为[14.16%,18.73%]；只反映当前64例的不确定性。

这只闭合约12.77%的baseline到AE参考误差差距。运动幅度代理0.057211→0.058039，
GT为0.071412；Laplacian锐度代理0.031284→0.031292，基本未变。不是动态质量突破。
已实际查看20例第4/6/8帧：0–7、10、11、24、34、36、41、44、45、48、58、60、63。
它们包含全部8个条件消融案例及数值上改善最大/最差案例。人腿重影、车辆形变、
背景结构漂移仍可见；局部改善不等于人物运动已经可靠。

两seed平均online：001车辆L1改善11.54%、004行人9.47%、007近景人物12.64%。
改善最大案例为034/044/001/045/007；变差为041/000/058/010/063/060。
保留全部案例，并单列已知AE异常041：其余63例baseline/oracle L1为
0.081568/0.076111，改善6.69%，结论并不依赖剔除041。

### 条件确实生效，但不能把敏感性当作质量增益

baseline条件注入严格0；oracle第2000步跨24rank平均注入RMS=0.09463，相对
输入RMS比例0.14326，presence=0.625即5未来/8主干slots。条件不是空值或死支路。
代码复核确认只输入未来geo128；未来texture没有进入条件，GT geo没有复制或
钳制到解码输入，所有未来C256仍由模型预测。

8个指定案例、seed101、online/EMA的correct/zero/shuffle全部完成、噪声匹配，
shuffle donor均为不同视频、同codec。online结果：

| 消融集 | baseline RGB L1 | correct | zero | shuffle |
|---|---:|---:|---:|---:|
| 全8例 | 0.109433 | 0.103942 | 0.144999 | 0.174503 |
| 单列041后的7例 | 0.076617 | 0.070227 | 0.124112 | 0.158501 |

全8例correct/zero/shuffle的texture MSE为0.249035/0.648251/0.827232。
正确条件texture为8/8更好，RGB为7/8；041在错误条件下RGB L1反而较小，与其
重建异常相符，不能说错误条件永远更差。

Oracle训练只提供正确条件，没有条件dropout。置零与打乱属于分布外干预，
证明网络依赖该条件，不能把zero模式等同独立训练的baseline，不能据此宣称
纯几何机制或WASD/位姿可控已成立。当前geo特征可带外观，且来自真实未来RGB。
这个oracle不是可部署的最佳生成模型，也不是数学意义的质量上限。

### 训练、完整性与下一判断

两组17个关键source/model/data/stats字段一致，4824个配对rank-step的noise-bin
计数一致，数据为同512窗口/498视频；未见坏shard日志。全部训练记录finite，
24rank均到2000。生产两步preflight后新进程继续训练成功；实际PT未下载，不能
独立核对完整optimizer/EMA/RNG、采样游标或位级曝光一致性。

最新DI约baseline1393、oracle1410未来token/s/NPU；进程累计峰值node0为
25.403GiB、其他节点22.025GiB，包含评估。两组进程耗时3.06h/3.18h，含IO与评估；
24卡idle guard覆盖约6.5h，全有pulse并正常stop，未见两小时空闲回收。55份发布
状态无errors，48份训练rank发布报告双目的地成功，没有实际OBS重新读取。

2944个现有sealed文件通过大小/SHA校验，3319个JSON正常解析；928个MP4正常
解码（920个九帧、8个四帧prefix），24个PNG正常。1180个有receipt的PT未下载，
本地实际PT为0；这不算已知损坏，但限制了对张量、权重和恢复的独立验证。

这次的有限正结果是：真实未来geo特征能帮助预测texture和RGB。它没有解决动态
画面质量，也没有隔离VGGT与DINO的差别。2000步是预算，不是收敛证据；前8例
oracle500/1000/1500/2000 L1=0.094410/0.091365/0.090161/0.089836，后段仍改善
但变慢。没有固定训练集自由生成评估，尚不能判定拟合能力还是泛化/覆盖瓶颈。

建议保留两组权重，暂不直接扩展同一pilot时长/全HQ或把oracle转述为可部署控制。
接下来的优先验证是最近讨论的DINO/VGGT在匹配视频RAE、decoder、数据与扩散
配方下的正对照，定位特征源与自定义压缩/texture组织的影响；这一后续未实现。

完整分析保存在 `D:/workspace/VggAE_DataAudit/stage58_complete_20261029/`：
`metrics/quality_metrics.json`、`metrics/bootstrap_relative_gains.json`、
`integrity/health_summary.json`、`integrity/summary.json`，以及20例
`visuals/caseNNN.jpg`。对照图包含EMA结果，主要量化表同时报告online/EMA，不能
把某张EMA图误标为online。用户原视频仍为RAW/AE/GENERATED或ORACLE/COPY四列。
