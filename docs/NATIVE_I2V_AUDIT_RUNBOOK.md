# 原生 I2V 与 R7 重放对照

入口为 `scripts/scale/48_audit_native_i2v.sh`。这是只读权重的推理验证，没有训练、优化器更新或新 diffusion 配方。保留原生 Wan I2V 的 VAE、图像/文本输入、patch embedding、输出 head、CFG 和 UniPC；不使用 `WanCompactAdapter`。

## 启动

继续使用原有 ModelArts 环境初始化，代码更新到包含 stage48 的提交。需要完整的 **Wan2.1-I2V-14B-480P 原生目录**，不能是 T2V、Diffusers 重排目录或只有 DiT 的分片。

先用原评估集前 4 个视频、两个原 seed 做首轮，共 8 个原生完整视频与 8 套比较：

```bash
NATIVE_CLIPS=4 \
WINDOW_NAMESPACE=r7_native_i2v_roundtrip_smoke_v1 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/48_audit_native_i2v.sh
```

脚本只自动寻找以下 I2V 目录，不使用公共配置中的 T2V 回退：

- `${VGGAE_REF_ROOT}/Wan2.1-I2V-14B-480P`
- `${VGGAE_REF_ROOT}/Wan2.1/checkpoints/Wan2.1-I2V-14B-480P`
- `/public2/LiZhen/yexiaoyu/ckpt/Wan2.1-I2V-14B-480P`

若权重位于其他已暂存目录，在启动命令前加 `NATIVE_WAN_CKPT_DIR=/实际目录`。若需从 OBS 暂存，设置 `NATIVE_WAN_URL=obs://实际模型目录`，可另设 `NATIVE_WAN_MIRROR_URL`；仅在集群复制，不向本机下载大模型。目前尚未核实当前 ModelArts 上的 I2V 权重位置。脚本会检查架构、索引引用分片、VAE、T5、CLIP 和本地 tokenizer；无法仅根据架构配置鉴别来源相同但训练分辨率不同的权重，输入必须明确为 480P。

首轮成功并检查画面后，完整 32 视频 × 2 seed：

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/48_audit_native_i2v.sh
```

默认完整 namespace 为 `r7_native_i2v_roundtrip_v1`。两组运行的首帧、seed、原评估索引规则一致，但完整组使用新的输出目录；不把 4 视频的契约改成 32 视频后继续 resume。

## 数据和 checkpoint

固定使用 `r7_domain_single_t2v2_legacy_diag_v2/eval` 缓存及 `configs/ae_reference_single_v1.json`。验证原训练 manifest、statistics、完整 32 视频顺序，再选其前 N 个。首帧来自已保存的真实 RGB，不重新随机截取，也不替换失败样本。

R7 AE 为 `r7_t2_c192_v2/joint/checkpoint_best.pt`，检查与已审计缓存一致的文件签名；保持 legacy 时序语义。原 diffusion 为 `r7_domain_single_uniform_reviewed_v1/checkpoint_final.pt`，内部必须 step6000、非 memory、无文本，并严格加载 EMA；不得自动取 latest 或其他实验的 best。

原生正 prompt 固定为 `A realistic continuous video of the scene.`，负 prompt 为原生配置默认值。无需未来 caption、轨迹、深度或目标视频；不自动扩写 prompt。训练数据中的真实未来仅用于离线参考和 R7 分支评估，不传给原生生成器。

两条生成分支都使用 seed101/211 加 `1009*原评估索引`，但 Wan 使用 NPU RNG，旧 R7 使用原有 CPU RNG，不能声称它们有相同初始噪声张量。

## 输出与判读

每个 case 保存：

- `native/clipXXX_seedYYY.mp4`：原生 81 帧、16fps 完整视频，max_area=480×832，保持首帧宽高比；40 步 UniPC、shift3、CFG5。
- `native/*_input.png`：实际首帧；`*_window.pt`：原生前一秒的 9 帧像素张量，索引 `[0,2,4,6,8,10,12,14,16]`，双线性抗锯齿缩放到 518×518 后 uint8 保存。
- `comparison/*_raw.mp4`、`*_raw_ae.mp4`、`*_native.mp4`、`*_native_r7_replay.mp4`、`*_r7_diffusion.mp4`：统一 9 帧、8fps，五路视频。
- `comparison/*_grid.png`：第 0/4/8 帧的紧凑无损网格；`*_windows.pt` 和 `*_latents.pt` 保留像素及 R7 中间表示，供进一步诊断。
- case JSON：指标、身份、文件 SHA256 和完成状态；`run_contract.json` 固定全部参数、代码、输入与模型签名；`summary.json` 为最终汇总。

原生输出首帧可能与输入有少量差别，记录 `native_anchor_l1`，不人为覆盖。原生 R7 重放的独立 anchor 从原生 frame0 编码；旧 R7 diffusion 使用原始缓存 anchor。这是两个不同来源的合法编解码链路，不拼接一个外来 anchor 制造重放误差。

重放与指标都读取同一份像素张量，不从有损 MP4 重新解码。MP4 使用 `macro_block_size=1`，不再把 518 自动缩放成 528；518 为偶数，适合 yuv420p。原生完整视频也不强行修改尺寸。网格仅展示三帧，持续运动应看 MP4。

若原生视频好、经过 R7 仍好，而 R7 diffusion 差，则说明 codec 能承载这批合理生成，但学习/采样尚未到达这些表示。若原生好、R7 重放明显劣化，需优先考虑 codec 的感知质量或域适配。原生重放好不能冒充 R7 diffusion 成功；与记录下来的真实未来不同，也不自动意味着原生生成错误。

`summary.status=completed` 只表示流程完整；`quality_passed=null` 明确需要人工查看。没有自动承诺生成质量。

## Ascend 执行与精度

首轮使用 leader 的 NPU0，串行处理 case；其余设备不加载模型。若原作业配置仍为 6 节点，其他节点由既有 TCP coordinator 保持存活，直到最终双写完成，不提前退出。建议单节点作业做首轮；此版本没有宣称多卡加速。

分三个 Python 进程执行 prepare、native、replay，确保 14B 与 StreamVGGT/R7 不同时驻留。T5 在 CPU；原生模型在每条视频结束后 offload。仅把 autocast 范围内的 Linear/Conv 参数保存为 BF16，time embedding、time projection、head、norm、modulation 保持 FP32；没有量化、裁层或新训练参数。

`utils/native_wan_runtime.py` 在隔离的 import namespace 中恢复上游 FP32 modulation，避免沿用旧 compact 适配的低精度残差修改；RoPE 使用 CPU FP64 构造相位、NPU FP32 实数旋转，避免 NPU complex 运算。注意力使用 NPU fused kernel，显式截取每个样本的有效 query/key 长度及 causal mask。原生 I2V 末尾的 CUDA synchronize 改为 NPU synchronize。

在加载大模型前，FP16/BF16 的小尺寸 attention 与 CPU FP32 参考进行数值验收，涵盖变长、causal 和缩放，失败即停止。这只验收兼容 kernel，不等于已经验证端到端 NPU 生成。具体参考：[Wan 上游模型](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)、[Ascend attention API](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000520.html)。

当前只有本地 CPU 单元测试与静态检查，实际显存、CPU 内存、加载耗时、NPU 数值与视频质量仍待本次作业确认。不保证与旧 178M 训练相同耗时；也不保证首轮不会暴露环境兼容问题。

## 双写、恢复与日志

输出位置沿用 `${OUTPUT_URL}/scale/${WINDOW_NAMESPACE}` 与 owner OBS `output/scale/${WINDOW_NAMESPACE}`；每 60 秒增量同步，结束再同步。源 R7/diffusion checkpoint 优先当前输出、再 owner 镜像；不可变 latent 数据仍读取其唯一的 owner cache。

每条原生视频完成后保存一次独立 receipt，不必等全部生成完。相同配置中断重启，添加 `RESUME=1`，同时保持 `NATIVE_CLIPS`、namespace、prompt 等不变。恢复检查契约和文件哈希，损坏的第一份文件可从第二份读取；未提交的 case 重做，已提交但两份均损坏的 case 明确报错，不当作成功。

例如恢复首轮：

```bash
RESUME=1 NATIVE_CLIPS=4 \
WINDOW_NAMESPACE=r7_native_i2v_roundtrip_smoke_v1 \
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/48_audit_native_i2v.sh
```

推理恢复以完整 case 为单位，不恢复到一个视频的第几步；没有优化器状态。若在 run contract 形成前就因缺权重失败，恢复要求尚不成立，应选新 namespace 重试。

控制台每个去噪步输出状态。原生 `DI_throughput` 单位为生成 frames/s，包含条件编码和 VAE 的单视频 generation 时间，不含初次加载/输出编码；对照阶段另记包含 IO 的新处理 frames/s。不会将重用的 case 计入新计算吞吐。完整墙钟可以由 launcher 日志与状态时间戳追踪。

每个阶段另存 `prepare_timing.json`、`native_timing.json`、`replay_timing.json`，记录本次尝试的起止时间及包含加载和 IO 的总耗时。已通过 11 项本地 CPU 测试，包括缺失分片/词表拦截、损坏副本恢复、注意力和 RoPE 数值检查、518×518 MP4 实际编码解码检查。

本机不下载 raw、模型或大中间结果；后续仅按需要把小日志与预览放 D 盘。
