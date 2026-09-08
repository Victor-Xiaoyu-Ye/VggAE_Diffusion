# R7 sampler 验证启动说明

**用户纠正后的定位：27 仅做只读接线检查，不是 diffusion 训练入口。25/26 是未完成诊断，不能作为本项目训练基线。此前 27→26 自动串联已撤销；`RUN_N1=1` 会明确报错。** 文件名为兼容已下发命令而暂时保留，不代表仍然训练 n1。本说明不能代替新的训练方案。

入口：`scripts/scale/27_validate_r7_then_n1.sh`。在现有 ModelArts 环境准备、代码同步和 StreamVGGT 权重 staging 完成后，替换原先的最后一条 scale 启动命令即可。无新增集群依赖；本机用于 CPU 测试的临时依赖不应复制进集群环境。

随附 `r7_sampler_validation_bundle.zip` 是增量包，解压到当前仓库根目录，保留 `scripts/`、`utils/`、`docs/` 层级；需要当前分支已有的 25/26 阶段及其依赖（审查基线 HEAD `c4459d2`），不是独立训练仓库。本地项目已经写入这些文件。

## 推荐启动

从仓库根目录运行：

```bash
FLOW_NAMESPACE=r7_sampler_validation_n1_probe_v2 \
  bash scripts/scale/27_validate_r7_then_n1.sh
```

也支持用脚本绝对路径从任意工作目录启动。第一次建议单节点任务：全部诊断只使用 node 0/device 0，没有 HCCL 集合通信。使用现有 6×8 配置也不会启动 48 份实验，其余节点退出，因而没有必要为此占满 48 卡。

默认顺序：

1. CPU sampler 单元测试。
2. 读取已训练 t1 geo112/tex80 AE 与历史确定性 n1 checkpoint。
3. 在 eval.csv 的第一个固定 clip 上重新编码；分别拟合 anchor/target 的训练统计量。
4. 检查归一化往返、oracle x0、确定性模型接入真实 sampler；使用 4 seeds、1/30/60 步、uniform/shift=3 两种网格；按与当前 flow 相同的 prefix-repeat-to-nine 方式解码 RGB。
5. 发布只读检查结果并结束。不会启动 25、26 或任何训练。

只验证链路，不训练：

```bash
RUN_N1=0 FLOW_NAMESPACE=r7_sampler_contract_only_probe_v1 \
  bash scripts/scale/27_validate_r7_then_n1.sh
```

已有 namespace 会拒绝重跑，重新验证请换名字。27 不接受训练 RESUME。不要将这个接线检查当作已建立的新 diffusion 训练方案。

v1 实机输出留下 `running/false`，日志最后到数据读取，没有完整失败原因。v2 保留旧数据并增加阶段记录（loading_codec / waiting_for_video / encoding_video / sampling 等）、SIGTERM/SIGINT 失败记录和 shell 退出码。默认 `CONTRACT_NUM_WORKERS=0`，仅影响单视频验证；26 的 worker 设置仍独立。整个作业被强制终止时 shell 也可能来不及落盘，不能由残留 running 判断进程还活着。目录冲突检查现在在视频枚举之前执行。

## 权重与输出

默认 R7：`r7_t1_c192_geo112_tex80_probe_v1/joint/checkpoint_best.pt`。

默认确定性模型：`r7_vggt_quick_geo112_tex80_v2/det_k1_n1/checkpoint_latest.pt`。该 namespace 来自历史日志；本机未验证 OBS 文件是否存在。默认先检查本地，再从当前输出根和持久镜像根拉取。若实际 checkpoint 存放位置不同，设置 `DET_CKPT_URL`、`DET_CKPT_MIRROR_URL` 或已有本地 `DET_CKPT`。R7 同理沿用 `R7_CKPT` / `R7_CKPT_URL` / `R7_CKPT_MIRROR_URL`。

输出位于既有 `$OUTPUT_URL/scale/$FLOW_NAMESPACE`，并沿用持久 OBS 镜像；若未设置 OUTPUT_URL，使用仓库配置的持久输出根。

优先查看：

- `contract/contract_status.json`：sampler/归一化/解码一致性，含当前 clip ID、窗口、物化 hash、权重签名和逐 seed/grid 误差。
- `contract/samples/`：RAW、AE、det direct、oracle sampler、det sampler PNG 对照。
- `contract/logs/`：验证日志。

## 如何读结果

`contract_status.passed=true` 只证明 sampler 接线正确。常量 oracle 和确定性模型均不依赖随机噪声，不能把它们算成 diffusion 生成成功。确定性 checkpoint 的历史格式缺少物化数据 hash，因此报告记录当前样本身份，但不保证与旧作业字节级同一输入；其 RAW/AE PSNR 单独报告，不作为接线 gate。

新 gate 要求 normalized endpoint 最大绝对误差 ≤1e-4、解码 RGB 最大绝对误差 ≤2e-3；另检查 anchor/target 往返和有限值。确定性模型维持历史 FP32 raw-latent 接口；默认 BF16 用于 frozen encoder 和 sampler 外层，不能因此声称 FP16/BF16 确定性网络本身经过质量验证。

`run_status` 表示训练预算/保存是否完成；`memory_status` 才表示随机噪声记忆质量。26 在质量未通过时仍可正常完成并输出诊断，这不是升级许可。fixed-path 成功也不等于新噪声成功。

若 contract 失败，只说明需要检查接线；若通过，也不能据此启动或认可旧 25/26 训练方案。新视频 diffusion 需要独立的模型、目标、数据与分布式训练设计。

## 本地验证范围

新增测试覆盖真实 sample_flow Euler、raw/normalized 接口、实际小型 SingleTargetGenerator 在外层 BF16 下保持 FP32 接口、错误 oracle 的负例与空矩阵拒绝。CPU 测试、Python 编译和 bash 语法检查用于验证代码；真实 checkpoint、视频、910B、MoXing 和 OBS 仍由此次集群启动验证。

已完成本地检查：新增 CPU FP32/BF16 测试通过；既有 `test_r7_flow_probe.py` 的公式、静态/CLI、CPU 张量、模型梯度、sampler、EMA 和指标回归通过；新入口 `--help`、Python 编译及 bash 语法检查通过。本机测试环境为隔离安装的 CPU PyTorch，未改变仓库依赖文件或集群环境。
