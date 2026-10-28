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
