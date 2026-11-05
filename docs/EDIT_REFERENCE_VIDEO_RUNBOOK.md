# Stage63：真实视频参考编辑与 RAE 写回

2026-11-05 实现。目标是直接交付真实17帧视频的完整对照，不再安排单独的合成
探针轮次。生产14B/910B效果和峰值显存尚需本次集群运行确认；本地完成的是
数学接口、真实小模型采样、源光流、恢复与编排检查。

## 启动

保留现有集群的代码与 `vggae_ref` 暂存方式。将本次代码同步到 OBS 后，在原来的
三台节点上执行同样命令。无需新增 bootstrap、下载新模型或重训AE：

```bash
REFERENCE_STAGE=smoke bash scripts/scale/63_edit_reference_video.sh &&
REFERENCE_STAGE=all bash scripts/scale/63_edit_reference_video.sh
```

默认3节点×8卡，24个既有dev源视频按rank分配。正式命名空间
`edit_reference_video_v1`，smoke 为 `_smoke`；两者数据/求解状态互不混用。
smoke仍覆盖24卡，把生成30步减到4步，RAE校准12次sweep减到2次。
smoke检查运行接口；没有质量指标门槛阻止其他case或正式运行。实际程序异常
仍以非零退出码暴露，不能把未完成作业当成功。

依赖 Stage60 已有 `vace_edit_pilot_v1` native bank 和生产 Scene C256/Stream 权重。
新作业先从本次输出/持久化两处读取缓存。原生 VACE 和 VAE 使用
`/cache/yexiaoyu/vggae_ref/Wan2.1-VACE-14B`。不加载额外T5（使用bank里的文本编码），
不要求下载新的光流模型。集群现有 `opencv-python` 用于CPU源光流。

## 四阶段的数据流

1. **anchors**：真实source、指令、合法源框 → 固定前端生成完整视频，取固定
   首帧作为共享参考。默认 `REFERENCE_ADAPTER=auto`，只读复用已完成Stage61的
   `region_sft/checkpoints/step_000750.pt`（RAW750）。每节点leader暂存一次，
   各节点核对hash；若任一节点无法恢复，则全体明确回退原生模型并记录原因，
   不让24个案例混用不同前端。两个后续方法使用同一张参考，不能用target挑帧。
2. **transport**：源视频光流＋参考 → RGB编辑提议 → 冻结RAE联合校准。
   使用相邻帧与首帧直接匹配、双向一致性和光度残差；先拟合合法提议，再用
   源对应同时约束传入/传出的编辑效果。校准每次只保留一个decoder反传图。
3. **render**：原生VACE原权重＋同一参考图、源视频、mask、seed、scheduler，
   分别输出 `native_reference` 与 `rae_writeback`。后者在实际scheduler最接近
   sigma=.3/.15的位置修改velocity，参考置信度低的区域交给当前native预测。
4. **summarize**：逐case核验两个方法的输入身份，生成同步三列视频和HTML图集；
   检查/修复两处媒体大小与SHA。没有完成的case单独列出，不静默取成功交集。

这轮主要对照是“同条件参考生成”与“加入完整RAE传输/写回”。尚未剥离普通RGB
投影或同VAE写回贡献，不能据此自动声称RAE独有收益；这些消融留待完整效果后做。

## 输入协议与当前能力边界

- 默认是文本 → 模型参考 → 视频。参考若改错/未执行，应记录为端到端前端失败，
  不把它全部归因于传输。模型参考外的内容用源图恢复，只将选定支持区域的变化传输。
- 现有四个人工source框可复用，其余case从模型/用户参考与源首帧的差分提取支持。
  这是变化支持，不是语义分割。当前没有声称集成SAM2；也没有依赖不稳定的旧Stream
  track head。光流是DIS，OpenCV无DIS或该调用失败时明确记录Farneback回退。
- 先做保持形状的外观编辑。源光流不能保证处理新物体、拓扑变形或所有遮挡；
  双向一致性不等于真实可见性GT。未知区域不硬拷贝上帧，写回权重会降低。
- 数据集的 `target` 不参与参考、支持、光流、校准、条件或checkpoint选择。
  当前汇总主要提供源变化、背景变化和实际写回强度诊断；低变化不等于编辑成功。

原生17帧/8FPS及尺寸继承缓存，保留native pad16/有效区域。RAE看0,2,…16九个
真实时刻，保持518letterbox；九时刻的flow单独计算，不能直接切片17帧flow冒充。
奇数帧在完整源光流提议上加源对应搬运的校准修正。恢复原分辨率时只传编辑差分，
加在原始source上，避免把全部源细节通过518缩放变糊。

VACE参考单独编码为latent前缀，和mask前缀一起进入采样；在因果VAE解码**前**
移除，不把参考RGB拼成第18帧。真实写回采用
`clean=x-sigma*v`，`delta=E(reference_mix)-E(D(clean))`，`v-=alpha*delta/sigma`；
alpha=.35，并限制每通道/时刻的clean增量RMS。此差分抵消无编辑时的VAE往返偏置，
不保证mask外像素经VAE感受野/后续去噪后严格不变。

## 参考与适配器选项

显式使用原生模型生成参考（不读取历史SFT）：

```bash
REFERENCE_ADAPTER='' REFERENCE_STAGE=all bash scripts/scale/63_edit_reference_video.sh
```

可用 `REFERENCE_ADAPTER=<local或OBS checkpoint>` 指定其他纯SFT/regionSFT RAW
适配器，配置默认从checkpoint读取，也可指定 `REFERENCE_ADAPTER_CONFIG`。
它只改变参考前端；最终两个renderer仍使用相同原生VACE。含RAE readout的训练
checkpoint不能装作纯SFT权重。更换前端/条件需新 `REFERENCE_NAMESPACE`，保留旧结果。

显式用户参考或独立模型参考支持 `REFERENCE_MANIFEST=<本地或OBS JSON>`：

```json
{
  "schema": "edit-references-v1",
  "cases": {
    "已有dev案例ID": {
      "image": "edited_reference.png",
      "provenance": "user_edit"
    }
  }
}
```

参考应是编辑后的源首帧/相同视场，`provenance` 允许 `user_edit` 或
`model_output`。相对图片路径相对原始manifest目录解析（含OBS），保留宽高比。
未提供参考的案例仍用固定模型前端；每例协议单独标注。用户参考不计入纯文本
指令跟随结论。不得将数据集edited target伪装成用户参考。

`REFERENCE_CONTROLS` 可换源框JSON；显式置空则全部使用参考差分支持。

## 保存、恢复与显存

输出两处：本次 `${OUTPUT_URL}/scale/edit_reference_video_v1` 与持久化
`output/scale/edit_reference_video_v1`。

- `manifest.json`：实际前端、权重/源码/参考身份、协议、选中案例。
- `anchors/<id>/`：共享参考PNG、支持PNG、源/参考/前端视频、可恢复payload。
- `transport/<id>/`：源状态、每sweep优化器checkpoint、RGB/RAE编辑参考视频。
- `render/{native_reference,rae_writeback}/<id>/`：实际生成视频、payload、写回事件。
- `comparisons/<id>/source_native_rae.mp4`：SOURCE / NATIVE / RAE，索引/FPS一致。
- `summary.json`、`media_sync.json`、`report.html`：覆盖、失败、输入匹配与双输出状态。

同命令可按case/阶段恢复；完成case会补发视频到新输出目录。仅重做渲染：

```bash
REFERENCE_STAGE=render bash scripts/scale/63_edit_reference_video.sh
REFERENCE_STAGE=summarize bash scripts/scale/63_edit_reference_video.sh
```

恢复smoke需指定 `REFERENCE_NAMESPACE=edit_reference_video_v1_smoke`；单阶段沿用
保存的有效配置。仅补汇总/双目录同步用summarize，不加载模型。
如先前transport失败，native仍可输出fallback条件基线；transport恢复后，renderer
会重新生成该基线，确保两方法共享同一mask，汇总也核对输入hash。

前端、Stream/RAE、最终14B分阶段加载；源状态编码后释放Stream再校准，不让
Stream/RAE与14B同时驻留。仍是一卡一案例，保持24卡并行、idle guard、协调器
心跳、DI_throughput及单位。DI包含各阶段对应的IO；没有未经实测的峰值显存承诺。
失败case保存原因并继续其他case/方法；最终汇总诚实区分完整和不完整。

## 验证与交付边界

本地测试覆盖原生reference前缀、真实tiny VACE/UniPC、编码差分写回符号与恒等性、
联合decoder求解/恢复、源对应方向、四阶段完整编排、target隔离、同条件比较、
双输出修复与恢复后fallback基线重生成。使用小模型或明确测试替身的部分不代表
生产14B画质；实际DIS检查也仅是受控运动工程验证，不是自然视频精度评估。
新旧相关测试共45项通过；实际DIS在纹理平移测试中央区域的平均光流误差约0.108px。
该数字是工程测试，不是自然场景benchmark成绩。Python编译、实际Bash编排和diff检查通过。
这一版没有执行新的集群作业。评审本轮应直接看完整视频与合法参考是否相符，
不要用求解loss下降替代成片质量。
