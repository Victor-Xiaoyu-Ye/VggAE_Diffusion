# R7 完整链条记忆测试与冻结子空间诊断

日期：2026-09-10。两个独立任务，均使用现有 t2v2 legacy AE 和缓存，不依赖 stage25/26，不重新训练 AE。

## 先启动记忆测试

在既有 ModelArts 6 节点×8 卡配置中使用：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/37_train_window_memory64.sh
```

默认 namespace：`r7_window_t2v2_legacy_memory64_v1`。6000 步总预算，其他主配方保持 shift3 无 aux：178M、batch96、lr1e-4、300步warmup、BF16、EMA。每500步评估并保存，64步 Euler，从纯噪声采样，四个固定评估seed：101/211/307/401。

训练来源只取 train manifest：固定种子的 shard 顺序中前64个不同 video ID，内容摘要和ID保存为 `memorization_subset.json`。这是方便复现的记忆集合，不声称随机代表全数据。标准化继续沿用原训练统计，避免同时改变表示。

64个样本驻留CPU内存；全局每64样本重新打乱，连续序列按rank取模分配，不丢epoch尾、不将少数固定样本永久分给某卡。DataLoader固定0workers，保留已消费批次回放及完整RNG恢复。每个rank核对所选数据内容签名。

训练缓存未存完整RAW：**记忆样本以冻结AE重建为像素对照，不伪称RAW重建。** 原heldout继续独立，全部ID参与重叠检查，16clips做RAW AE gate与自由生成评估。没有用heldout训练或拟合PCA。

每次评估完整矩阵：64记忆样本＋16heldout，online/EMA×4seed，共640行。64记忆样本全部保存预览和latent；PNG是原分辨率证据，MP4可能由兼容编码调整尺寸。数据行明确标注 `split=memorization/heldout`。新增逐帧L1、latent低/中/高频误差、评估耗时；原噪声分桶、梯度、内存、DI吞吐保留。

`checkpoint_best_reconstruction.pt` 在此实验按 **memorization EMA L1 vs AE** 选择；config明确记录选择split，不把它当heldout最佳模型。评估输出较旧实验大得多，训练步计算规模基本不变，总时长不能沿用旧估计。

建议先看500与1000步：所有训练样本是否开始恢复主体/轮廓和连续运动；不能只看loss。记忆通过不等于泛化通过。6000是预算上限，不自动续跑更多步。

计划暂停可在同一6000步调度下设置：

```bash
STOP_AFTER_STEPS=1000 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/37_train_window_memory64.sh
```

继续同一实验：

```bash
RESUME=1 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/37_train_window_memory64.sh
```

world size、数据内容、配方与评估协议必须相同；不是从旧shift3/aux checkpoint续训。完全重跑使用新的 `WINDOW_NAMESPACE`，不要覆盖旧目录。双目的输出、checkpoint读回退、每60秒增量发布、结束时发布回执都继承stage29；运行失败仍保留状态与已写结果。

## 独立启动子空间诊断

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/38_diagnose_window_subspace.sh
```

默认 namespace：`r7_window_t2v2_legacy_subspace_s2500_v1`。读取旧无aux shift3 的 `checkpoint_best_reconstruction.pt`，强制内部step2500及EMA完整性检查；不使用最新checkpoint替代。不需要等待记忆测试结束，建议单独排队以免争抢同一批48卡。

16heldout×2seed×12arms=384行：generated、target、三个频带修正、PCA前8/32方向修正，以及每项的匹配剩余MSE对照。PCA在64个训练视频上拟合并保存ID、特征值和basis；FFT/PCA诊断计算在CPU上执行，避免假定Ascend支持相关算子。

修正为 `generated - P(generated-target)`。对照为沿原误差所有方向均匀收缩，使每个future slot的剩余MSE与修正结果相同。它区分“只是减少了更多误差”和“特定误差方向更重要”。所有修正使用GT，是**离线oracle，不是可部署生成结果**。

保存全部latent输入、所有16个视频的预览、逐帧误差、每rank指标及最终矩阵完整性检查。DI是采样token-steps/s，解码耗时另列；不同arms重复记录同一次采样时间，不应累加。先检查 `ae_gate.json`、`summary.json.complete`、384行与双写发布状态，再比较频带/PCA修正和匹配对照。

## 本地验证与限制

CPU合成测试覆盖正式trainer记忆模式全状态中断恢复、48rank非整除样本分配、subset内容指纹、PCA、频带分解、匹配MSE和oracle完整输出；既有flow/aux/双写失败注入回归一并运行。shell静态检查和Python编译检查应通过后推送。

CPU结果不代表真实视频质量、NPU前向、HCCL或OBS已经验证。两个任务都不保证产生可用视频；它们为下一次架构/数据投入提供可判定的证据。
