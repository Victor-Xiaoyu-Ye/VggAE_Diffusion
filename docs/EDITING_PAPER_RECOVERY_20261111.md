# Stage69 首轮缺项与最小恢复

2026-11-11。输入是用户本机 `D:/中转站/summary.json` 和
`D:/中转站/media_sync.json`；尚未取得本轮全部视频或失败案例的 prepared tensor，
本次是运行失败诊断，不是新增画面质量结论。

## 已确认的运行状态

- 固定源集合 123 个，目标上限128；暴露源指纹筛查2076/2076完成。
- 新无状态 SFT 完成1500次更新，skipped=0；converged=false，即用完既定更新预算，
  不能称为已验证收敛或证明画面质量。
- 预期1277条最终采样，实际1266条；所有成功项均已有自动评价。
- Native 155/155、Native+reference/SFT/RGB base/RAE base各123/123完整。
- RGB full、RAE full各152/155。5项RAE消融各31/32；其余消融完整。
- 媒体核对3321项，media_failures=0。尾部报错不是双写同步失败。
- 人评导出93/96，缺3组源于同一条RAE视频缺失；尚未收集真实人评。

所有11条采样失败均为 `ValueError('Edge confidence must be in [0,1]')`：

| 源ID | 缺失输出 |
|---|---|
| `372f632ee36fa3979f05af7b` | RGB/RAE full各seed42和43；RAE reinject、no_temporal、shuffle_graph、bridge_only、velocity_only各seed42，共9条 |
| `5ec29e5cd63933637cf1fda3` | RGB/RAE full各seed42，共2条 |

总计26条summary失败记录包含覆盖率、采样、评价、6个比较视频和人评的连带缺项，
不是26次独立采样故障。no_graph与no_reliability两项通过也符合边置信度路径的定位。
7条keep指标为空，评价代码会在目标差分覆盖全画面、没有保持区域时返回null；
这与采样失败是两种情况，不能填零冒充更优。

原始报告SHA256：

```text
summary.json    6e1c924a6df6a994f72dd9d791ca036c2892fd0e7480053ddc4f59ac2f04ef78
media_sync.json 1c027d8ef84eec7331be4a6aaff85972c7a94a7212edc59b58d59ef9926a967a
```

## 数值缺陷与修复边界

本地通过真实 `_stencil` / `_apply_stencil` / `propagate_reference` /
`build_source_edges` 路径复现：常数图像在合法亚像素对应下，四个FP32双线性权重相加
可以得到 `1.0000001192092896`。此前求解器严格拒绝任意大于1的边置信度，导致
同一源在不同seed及多条反馈臂上全部失败。测试只固定光流估计值，未伪造置信度。
目前还没有读到生产失败tensor的实际min/max；新日志会明确记录它们。

修复只在求解器入口接受 `8 * float32.eps = 9.536743e-7` 内的边界舍入误差，
并投影到[0,1]，记录src/dst/min/max/改动元素数。合法数值保持原值，缓存tensor
不被原地修改。NaN、Inf和实质越界继续失败并打印实际范围，不能无条件clamp掩盖坏数据。
反馈事件和中间snapshot保留此诊断。有效输入的求解行为未重新调参。

## 现有三节点任务的恢复

更新代码与OBS代码副本后，三台节点仍执行同一个入口：

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
PAPER_STAGE=repair bash scripts/scale/69_complete_editing_paper.sh
```

保留 `editing_paper_evidence_v1`，本次不用新namespace，也不再次执行smoke→all。
`repair` 从旧manifest读取原配置，在新机器恢复pilot数据、AE与选中权重，然后按持久化
phase记录只生成缺失/失败案例。成功记录和样本保持，未完成单例从相同seed重新采样。
之后补评价、比较视频、人评导出及两端媒体校验；本轮新增采样预期为11条。
加载权重与准备IO仍有成本，不能把11条采样时间当作整个任务耗时。

不运行重新选集、前端anchor生成、state/native缓存重建或SFT。成功记录复用不计入新生成
DI_throughput。双读双写、CPU阶段隔离、idle guard和日志发布沿用现有入口。

`configs/editing_paper_resume_compat.json` 仅允许本次数值修复前后两个文件的确切SHA
迁移；其余代码、配置、输入或权重变化仍拒绝混用。原manifest/contract/生成身份保留，
`recovery/compat_node*_rank*.json` 记录迁移，新采样另记runtime_code。没有关闭代码校验。

完成标准：coverage各臂补满1277、failures及media_failures为空、人评包96/96、summary
status=completed。可用指标分母必须照实报告；无保持区域导致的null不是质量门槛。
若新日志显示置信度超过舍入容差，保留失败记录并检查生产tensor，不能继续扩大容差。

## 验证

60项相关本地测试通过：求解器20、论文数据/生成/恢复10、真实Bash入口5、反馈/媒体/
VACE接口25。实际小模型端到端覆盖只补失败seed、成功payload字节不变、无额外SFT、
全部17臂、双端媒体恢复、坏数值拒绝，以及未知代码/配置改变仍拒绝旧contract。
5个改动Python文件编译和Bash语法检查通过。

本地使用明确的小模型fixture；没有把它称为14B/910B/HCCL/生产OBS验证。
修复后生产补跑及最终效果仍待用户启动、回收结果。暂不追加新研究实验。
