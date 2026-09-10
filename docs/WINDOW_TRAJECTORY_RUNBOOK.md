# Memory64 采样轨迹诊断

启动（沿用ModelArts六节点×八卡配置）：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/39_diagnose_window_trajectory.sh
```

只读加载 `r7_window_t2v2_legacy_memory64_v1/checkpoint_final.pt`，内部必须step6000。
输出namespace为 `r7_window_memory64_trajectory_s6000_v1`。不要加载旧一万视频的checkpoint。
AE仍为t2v2 legacy，签名、统计、训练manifest、64样本内容摘要与heldout独立性均核查。
heldout RAW AE门槛保持23.5dB。诊断没有optimizer，不写训练checkpoint。

16记忆样本为原64集合的前16个，16heldout为原评估集合，online/EMA各两个seed101/211。
噪声沿用stage37的样本index：heldout偏移64，避免换噪声导致不可比。
使用正式 `WindowFlow.sample` 64步Euler，只读observer保存第0/8/16/32/40/48/56/60/63步。
每个节点记录自由采样状态、当前x0、相同u与初始噪声的GT线性加噪x0，另记录最终结果。
GT诊断在完整自由采样结束后执行，不将GT或诊断输出反馈给轨迹。

每个节点均解码并保存latent MSE、RGB L1、逐帧L1、运动与解码耗时。
state在u>0本来含噪，其解码不能当作最终生成质量；应主要比较沿途x0与最终结果。
GT probe是离线oracle，不能称为可部署结果。
默认32样本×2权重×2seed×28arms=3584行。

保存顺序：每样本先写 `latents/`（包括所有节点、最终latent、target、anchor和统计），
再逐行写 `metrics_rank*.jsonl`；全局矩阵完整后写 `metrics.jsonl` 和 `summary.json`，
之后才生成少量PNG整段帧对照。每split前2个视频，online/EMA×2seed，共16张图。
不编码MP4、不导出每帧独立PNG；后续可从已存latent重新生成视频。
`summary.json.complete=true`表示指标完成，`previews_complete.json`单独表示预览完成。
DI为诊断forward token-evals/s，包含64采样+9GT probe与CPU快照开销，不与正式训练DI直接比较。

双读checkpoint回退、双目的增量发布、失败状态保留继承stage29。新运行要求空namespace；
该只读诊断不支持恢复半次运行，若需重跑使用新的 `WINDOW_NAMESPACE`，不要覆盖已存结果。

验收顺序：AE gate → 3584条唯一指标 → 记忆/heldout、online/EMA分别比较
`x0_00`（纯噪声一次预测）与`final`，再找首次明显恶化的节点。
若u1单次解码已清晰而后续恶化，才针对采样轨迹/训练状态覆盖设计干预；
若单次latent误差小但RGB仍差，继续检查decoder敏感方向，不仅凭MSE结论。

本地CPU测试覆盖observer与正式采样终点逐元素一致、u1两种probe一致、完整指标/latent输出，
并回归已有训练resume/双写/flow测试。真实NPU、HCCL与OBS必须由本次集群运行验证。
