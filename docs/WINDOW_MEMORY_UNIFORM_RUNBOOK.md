# Memory64 均匀噪声时间对照

启动stage40，沿用6节点×8卡：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/40_train_window_memory64_uniform.sh
```

输出：`r7_window_t2v2_legacy_memory64_uniform_v1`。
**从头训练，不从旧memory64的6000步接着训练。** 对照是stage37完整6000步结果。

唯一训练干预是噪声时间分布：原 `sigmoid(N(0,1))` 再shift3，替换成未经warp的 `U(0,1)`。
代码要求uniform配合time_shift=1，否则报错，避免声称均匀却再次warp。
端点继续夹到[1e-5,1-1e-5]，不改变原loss中的floor=.05。
AE、64训练视频选择、全部训练统计、模型宽度/深度、初始化seed、batch96、lr1e-4、
warmup300、6000步cosine、EMA、无aux、正式64步Euler采样等均沿用stage37。
均匀时间会改变加权loss的有效区间贡献；没有importance补偿或额外loss归一化。
它验证整个时间抽样配方的效果，不能声称单独隔离了采样密度与梯度权重两个机制。

保留5个噪声区间的sample_fraction、loss_fraction、x0_mse/objective、全局gradient norm。
均匀采样长期每bin占比应约20%，短日志窗口会随机波动。训练loss与旧实验不是相同分布下的量，
不要单凭train/loss更低选赢家；比较固定eval seed、固定u的probe和正式自由生成。

每500步评估：64训练样本+16heldout、online/EMA、4seed，共640行，保存全部64训练样本预览/latent。
训练缓存无完整RAW，记忆RGB误差对照AE；heldout使用RAW AE gate。最佳checkpoint仍按记忆EMA L1选，
不得报告成泛化最佳。DI、内存、评估耗时、中间checkpoint、完整恢复、双读双写沿用stage29。
训练步和输出矩阵规模不变，实际运行时间仍以910B实测为准。

计划暂停而保持6000步scheduler：

```bash
STOP_AFTER_STEPS=1000 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/40_train_window_memory64_uniform.sh
```

同一实验继续：

```bash
RESUME=1 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/40_train_window_memory64_uniform.sh
```

不同抽样配方/数据/world-size不允许恢复；旧logit-normal checkpoint保持原contract兼容。
新跑用新namespace，不覆盖旧结果。

6000步完成后独立启动stage41，同stage39协议检查新checkpoint：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/41_diagnose_window_uniform_trajectory.sh
```

读取uniform实验的checkpoint_final.pt，强制step6000。输出
`r7_window_memory64_uniform_trajectory_s6000_v1`，先存3584条指标及latent，再少量PNG预览。
不要提前起stage41，也不要在只有1000步checkpoint时用latest伪装final。

验收：完整采样应保住中段x0画面质量，尤其u=.375→.25不再明显恶化，同时检查高噪声预测
是否因新配方变差。若低噪声修复但高噪声严重退步，则不是整体通过。heldout单独评价，
不承诺修好尾段就解决泛化。先看500/1000步趋势，完整结论用同预算与轨迹对照。

本地测试覆盖旧分布抽样与contract不变、uniform覆盖及非法warp拒绝、正式loss/sampler不变、
uniform完整训练恢复与跨flow拒绝，以及原有双写/轨迹回归。真实NPU和生成质量待运行。
