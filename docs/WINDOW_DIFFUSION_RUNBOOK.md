# 完整窗口 R7 diffusion：启动与结果判读

本版独立实现新 DiT/trainer，不基于25/26，也不复制V-RAE setting。目标是首帧条件下联合生成其余8帧；默认先用历史 `r7_t2_c192_v2/joint/checkpoint_best.pt`，另用 `AE_VARIANT=t1geo112` 比较。

**v2修复：v1已经在48卡910B/BF16运行，但t2v2的AE回放只有18.73dB。历史t2v2使用跨时间GroupNorm，当前逐帧GroupNorm虽然键名相同，却不复现历史AE。v2显式恢复legacy语义，并增加两次实际重建准入检查。新缓存和新训练命名空间不能混入v1，不能拿v1 diffusion checkpoint原样续训。真实修复效果仍须集群回放，代码通过测试不代表已恢复24.5dB。**

## 直接启动

沿用之前 ModelArts 的仓库、Python/torch_npu、StreamVGGT与source dual-AE输入 staging。以下命令在每个节点执行同一份；通常6节点×8卡。脚本可从任意工作目录调用。替换示例中的仓库路径即可。

```bash
# 第一轮明确做无文本 I2V，免去额外 UMT5 依赖；首帧仍是条件。
AE_VARIANT=t2v2 NO_TEXT=1 \
WINDOW_NAMESPACE=r7_window_t2v2_legacy_x0_diag_v2 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/30_run_window_baseline.sh
```

30先执行31的AE A/B回放，再执行28缓存(train/eval)、merge和29训练。31从旧 `r7_window_t2v2_diag_v1/eval` 缓存读取16个完整RAW片段，重新编码、解码legacy/framewise两种模式，不使用旧latent来冒充正确重建。默认legacy平均PSNR>=23.5、比framewise至少高2dB才继续；结果和图像双写到 `scale/window_ae_audits/$WINDOW_NAMESPACE/attempt*/`。失败即停止，不自动降门槛。可用 `AUDIT_CACHE` 指定旧eval缓存根路径。

新缓存默认 `r7_window_t2v2_legacy_diag_v2`，显式记录 `window_codec_runtime`。默认1个缓存partition覆盖48卡，train9935/eval64由实际OFT文件清单决定。条件单独编码首帧，eval保留完整原片。新缓存创建完成后，29在任何训练更新前再检查其真实AE回放PSNR>=23.5。t2的完整未来联合生成允许未来内部共享统计，首帧条件始终独立获得。

默认训练：768宽/12层/12头、batch1×accum2×48=96，BF16、AdamW、LR1e-4、warmup300、总预算6000步；x0输出、截断加权损失、logit-normal噪声时间shift1、64步Euler。每500步保存与评估，每10步日志。预算是起点，不是“6000步必然出好视频”的承诺。

上面的命令在两次AE检查通过后直接训练6000步。如希望先暂停50步，可保持同一模型、完整未来窗口和6000步scheduler：

```bash
AE_VARIANT=t2v2 NO_TEXT=1 STOP_AFTER_STEPS=50 \
WINDOW_NAMESPACE=r7_window_t2v2_legacy_x0_diag_v2 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/30_run_window_baseline.sh
```

这是同一训练任务的健康检查，不是n1预训练；50步的生成质量不能作为最终结论。确认运行健康后完整恢复到6000：

```bash
AE_VARIANT=t2v2 NO_TEXT=1 RESUME=1 \
WINDOW_NAMESPACE=r7_window_t2v2_legacy_x0_diag_v2 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/30_run_window_baseline.sh
```

恢复仅限同一个v2训练，跳过31和已完成缓存准备。除 `STOP_AFTER_STEPS` 和输入/输出定位外保持模型、目标、归一化、数据、精度、batch、world size、scheduler总预算和评估配置一致。不同实验用新namespace：例如 `AE_VARIANT=t1geo112 WINDOW_NAMESPACE=r7_window_t1geo112_framewise_x0_diag_v2`，或 `PREDICTION=velocity WINDOW_NAMESPACE=r7_window_t2v2_legacy_velocity_diag_v2`。

若要caption条件，去掉 `NO_TEXT=1`；30会选择性调用15预计算UMT5 sidecar，29按节点暂存其显式索引文件。已缓存的空/缺失caption没有逐ID证明，故本版严格拒绝缺失ID，不会悄悄用空文本补齐。可通过 `TEXT_DIR` 指定完整sidecar。首轮无文本与后续有文本必须分开标记，不能直接归因于表示改进。

## AE权重选择

`checkpoint_best.pt` 是默认候选文件名，不证明它恰好是日志PSNR峰值权重。默认t2v2历史峰值约24.57，t1geo112约24.62。启动打印实际step与签名；在任何训练更新前，使用同一held-out缓存解码并写 `ae_baseline.json` / `ae_baseline/`，记录该真实权重的重建PSNR和RAW/AE/mean/copy对照。缓存、权重、encoder/source签名不一致会报错。

如果已确认某个周期checkpoint才是所需最优版，通过 `R7_CKPT`、`R7_CKPT_URL`、`R7_CKPT_MIRROR_URL` 指定它，同时使用新的 `R7_CACHE_VERSION` 和训练namespace，不能把旧统计嫁接过去。各AE的完整同集排名仍需实际集群回放。

28沿用12的数据/编码基础设施，但新缓存显式记录 `independent_anchor=true`，condition仅由首帧编码。缓存保留整段anchor与独立anchor的相对L2差异用于审查。t1生成8个latent时刻，t2生成4个；都一次解码9帧。历史的0.95几何代理门槛不用于阻止本轮用户已接受的AE对照；旧gate与旧产物不变。

## 双写、双读与中间结果

- 输出写 `$OUTPUT_URL/scale/$WINDOW_NAMESPACE`，并写持久owner OBS `output/scale/$WINDOW_NAMESPACE`。沿用 `scripts/spatialvid_config.sh` 的根路径。
- AE和恢复checkpoint按当前输出→持久镜像回读；恢复必须显式 `RESUME=1`，两处都没有checkpoint就失败，不会从零冒充续训。可指定 `RESUME_URL` / `RESUME_MIRROR_URL`。暂存使用partial→rename。
- 每60秒逐文件增量同步两处；使用稳定快照，不传尚未完成的临时checkpoint。每个目标独立记账，一处失败仍尝试另一处；失败会重试并落 `publication_status.json`。结束时再同步并检查，两处有任一最终失败就非零退出。该状态表示copy调用完成，不是OBS服务端校验和证明。
- checkpoint最新、周期、final/paused均含模型、EMA、optimizer、scheduler、scaler、每rank RNG、已消费batch数、统计和不可变训练契约。昂贵视频评估之前先保存周期checkpoint。
- 数据恢复以固定零worker、私有shuffle RNG重放已消费batch后恢复训练RNG；CPU已测逐张量精确一致。长训练重放有I/O成本，不是O(1)恢复。超过恢复checkpoint的日志行被保留到 `history_before_resume/` 后从活动JSONL剔除。
- 训练产物包括 `metrics.jsonl`、`eval_samples.jsonl`、`tb/`、`samples/stepXXXXXXX/`、每rank状态、NPU日志、launcher退出码。保存online/EMA、多个固定seed、生成latent和PNG/MP4对照；MP4编码失败仍保留PNG并在预览manifest标注。
- 评估按clip分配给多个rank，rank0汇总所有指标，避免单卡串行评估让其余47卡长时间等待。rank0所在节点预览在主目录；其他节点的文件在两处输出的 `workers/nodeN/` 下，同一clip有唯一编号。
- `checkpoint_best_reconstruction.pt` 按EMA与AE的RGB L1选取，仅是重建距离指标，**不是最佳生成质量**。自由采样视频、运动与多样性仍需查看周期checkpoint。
- v2生成评估的 `rgb_copy` 是真正重复RAW首帧。原来的anchor-latent-repeat仅在AE回放中标为 `latent_copy_diagnostic`，不能当静止RGB视频基线。指标另含RAW运动和RGB复制相对RAW的距离。
- `DI_throughput` 明确为每卡未来latent tokens/s（不计condition token），使用设备同步后的最慢rank更新耗时；还记录全局clips/s、global batch、累计曝光、梯度norm、LR、峰值显存。评估/保存时长不混入训练step吞吐，实际成本还应结合状态时间戳和总运行时长。

如果未走30，可分别运行28的 `MODE=train/eval/merge`，再调用29。缓存是持久数据，仍放 `cache_latents/$R7_CACHE_VERSION`；不是把整套视频或latent复制两遍。训练中间产物才进行双写。旧数据输出不会自动删除；已有训练namespace必须显式恢复或换新名称。

## 已验证与未验证

本地CPU测试覆盖FP32/BF16 oracle采样、t1/t2形状与条件/文本mask/梯度、完整合成训练与暂停恢复、双目标增量写入/一处失败仍写另一处/恢复重试、输入fallback与原子替换。合成decoder仅服务测试，禁止用于真实训练。

v1日志已证明48卡BF16训练、周期采样与恢复实际执行，各节点双写状态成功；其AE兼容性问题使生成质量结论无效。v2新增CPU验证覆盖历史GroupNorm数学等价、权重键无法识别运行语义、新旧缓存/恢复契约拒绝和PSNR准入失败。修复后的真实AE质量和新生成结果仍待集群验证。先看AE回放是否维持原有重建，再看500/1000等固定留出样本是否有内容、条件一致性和合理运动。

研究差异性目前来自“重建相近、时间压缩与几何保留不同”的AE对照；本版未加入新的几何关系损失，也未证明几何收益。先训练出可靠基线，再据失败机制建立方法贡献。
