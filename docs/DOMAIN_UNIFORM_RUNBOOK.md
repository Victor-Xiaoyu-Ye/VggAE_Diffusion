# SpatialVID 单域 / 匹配混合域训练

状态：v1实机缓存约12%时因选中OBS对象缺失停止，尚未进入DiT训练。v2已在完整HQ目录清单基础上重选并加入模型加载前对象检查；新v2实机运行待验证。继续使用冻结r7_t2_c192_v2/joint/checkpoint_best.pt、legacy codec、独立首帧、uniform时间分布、原weighted x0 loss和178M DiT。每组全新训练，禁止从memory64完整恢复。

2026-09-11只列举文件路径和大小，未下载原视频：HQ展开视频根360610个非空MP4，metadata365362行，4752行在该根无对应对象。旧single训练缺35、mixed缺41，两个eval各缺2。v2在实际存在文件中按原规则重选：core候选4333，other11945；六个split共4096个唯一ID全部存在。原始HQ与旧oft10000子集保持区分；这里只核对指定展开路径，不断言缺失片段在其他归档中也不存在。

## 启动

继续用既有ModelArts环境初始化和6节点×8卡启动方式，所有节点执行相同命令。先跑单域：

```bash
ARM=single RESUME=0 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/43_train_domain_uniform.sh
```

匹配混合域在另一作业运行：

```bash
ARM=mixed bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/43_train_domain_uniform.sh
```

默认包含缓存准备。只检查缓存可运行stage42；缓存已完整校验后可PREPARE_CACHE=0跳过编码，入口仍校验缓存合同。两组不要共用同一OUTPUT_URL和本机工作目录同时执行，按两次集群作业运行。

```bash
ARM=single bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/42_prepare_domain_cache.sh
```

阶段性暂停仍保留6000总预算和原学习率调度：

```bash
ARM=single STOP_AFTER_STEPS=1000 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/43_train_domain_uniform.sh
ARM=single RESUME=1 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/43_train_domain_uniform.sh
```

RESUME=1只限同arm/namespace恢复；沿用stage29模型、优化器、scheduler、EMA、RNG与消费游标恢复。数据流恢复会重放消费游标，可能有I/O等待。默认namespace分别r7_domain_single_uniform_v2、r7_domain_mixed_uniform_v2；新起重复实验需新WINDOW_NAMESPACE，不能覆盖已有训练。此次v1→v2须RESUME=0，不复用v1部分缓存，不手动指定旧v1 WINDOW_NAMESPACE。

6000结束后，对其他室外验证域做只读诊断（不是封存test）：

```bash
ARM=single bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/44_diagnose_domain_other.sh
ARM=mixed bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/44_diagnose_domain_other.sh
```

自定义训练namespace时相应设置DIAGNOSTIC_SOURCE。stage44加载final6000/EMA，原训练统计保持不变；新eval仅在CSV hash与AE表示/RAW/独立anchor合同通过时允许。默认诊断仍严格要求原eval，不放宽旧实验。此标准诊断包含多个采样/条件对照，耗时高于单次采样。

## 数据与缓存

配置configs/spatialvid_domain_v2.json冻结源CSV SHA256、OBS目录快照SHA256和六个split的有序ID；v1配置保留作历史。集群读取dataset/SpatialVID-HQ完整metadata，拒绝内容变更；使用真实视频根videos/SpatialVID/videos，不再默认跳到oft稀疏镜像。不下载全量视频，只在集群按需暂存选中片段。本机仅CSV和对象目录信息。

每组2048训练片段、每片段4个分散1秒/9帧窗口，共8192训练样本；每个窗口独立AE编码，保存window_id、frame_indices。两组共享512街景，其他1536条匹配替换；试验测量目标域集中训练的效果，不单独证明混合训练负迁移。来源YouTube映射缺失，当前只有clip-ID隔离。

缓存命名：owner OBS cache_latents/r7_domain_{single|mixed}_t2v2_legacy_diag_v2/下train、eval与other/eval。每arm eval128条街景，other/eval128条其他室外；test清单仅物化，不编码不训练。

每次stage42在加载AE前逐个检查本arm训练及两个eval共2304个唯一对象。最多8并发/节点，访问异常重试，缺失与访问异常分别记录object_preflight.json并双写。失败不替换、不启动编码。快照存在性不保证未来可读或视频能解码，后续缓存仍保持零失败门槛。

缓存使用历史fp16编码设置、64样本/tar、48rank分片。已有缓存按原world size/config与CSV hash验证后复用。解码失败立即停止，不替换到其他片段、不把失败计入已提交游标；可重试同cache恢复。缺失源视频持续失败时需重新设计两组清单和新namespace，禁止直接跳过。缓存合并要求零失败、8192/128/128精确数量、至少48训练tar。

完整latent缓存沿用既有owner OBS持久化机制，不在OUTPUT_URL复制整套latent。缓存进度/分片保存在该缓存根；准备日志与清单在scale/domain_cache_{arm}_v2双写至OUTPUT_URL与owner output，最终双写失败返回非零。

训练输出沿用stage29每60秒双写、每500步checkpoint与预览、双源checkpoint读取、失败/暂停/完成状态。每步记录DI throughput、耗时、峰值显存，噪声分桶与中间评估保留。准备和训练均保存数据清单身份；凭据不在仓库，集群使用已有MoXing权限。

## 预算与评估

两组各6000steps/global batch96，约576000窗口曝光、70.3次/窗口；并非新增70倍独立数据。每500步评估固定32街景clip×2seed×online/EMA，共128行，8个clip预览。原训练器AE gate23.5dB保留；新域若未通过，先查AE实际表现，不自动降低门槛。其他域由stage44检查，封存test尚未提供入口以免被反复调参使用。

新缓存必须实机测耗时，不能保证总时长与旧memory64相同。DiT预算相同，但恢复重放、流式数据及缓存时间不同。比较两组时同时报告域内误差、感知表现、运动、首帧复制对照和AE上限；尾段退化未被本实验自动解决。

## 已验证范围

- 25项CPU单元测试（对象预检、数据清单/缓存门槛及原flow/resume/diagnostics等），含缺失/访问异常、变更CSV、跨split ID重叠、缺RAW、错窗口数、少样本/少shard的拒绝。
- 两组16384条预计算窗口索引与生产_compute_frame_indices逐条一致。
- 完整CSV在新物化器重建六个split，固定顺序和数量；Python编译、五个相关shell bash-n与diff检查通过。
- NPU编码、OBS缓存与新的alternate-eval路径仍待集群验证；不能将本机检查当作已生成视频。
