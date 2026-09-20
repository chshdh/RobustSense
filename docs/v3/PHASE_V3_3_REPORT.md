# RobustSense V3-3 阶段报告

## 1. 完成状态

V3-3 已完成 15 个用户分组连续风险单元；P2 主模型未重新训练。

- 协议哈希：`efb672880ee0ff1680cafed70e33ce265ddcc65e24e6b7c44c7798d6500b08df`
- 正式单元：15/15
- validation 选择 output-compact：4 个单元
- validation 选择 full-compact：11 个单元

## 2. 五折聚合结果

| 选择器 | AURC↓ | 错误检测 AUROC↑ | AUPRC↑ | Risk@80%↓ | Macro-F1@80%↑ |
|---|---:|---:|---:|---:|---:|
| `grouped_risk_full_compact` | 0.339405 | 0.707010 | 0.830246 | 0.399818 | 0.616438 |
| `grouped_risk_output_compact` | 0.345812 | 0.713224 | 0.837259 | 0.401978 | 0.615680 |
| `grouped_risk_selected` | 0.340077 | 0.707375 | 0.831094 | 0.399898 | 0.616157 |
| `mean_binary_certainty` | 0.345228 | 0.727265 | 0.843617 | 0.405033 | 0.611600 |
| `p2_system_reliability` | 0.436400 | 0.508706 | 0.706046 | 0.441397 | 0.580626 |

## 3. 预声明主比较

### grouped_continuous_risk_gain

- 候选：`P2/grouped_risk_selected`
- 参照：`P2/mean_binary_certainty`
- AURC 差值：-0.005150；候选胜出 3/5 折。
- 错误检测 AUROC 差值：-0.019890；候选胜出 0/5 折。
- 双主指标判定：**未同时通过**。

### compact_sensor_increment

- 候选：`P2/grouped_risk_full_compact`
- 参照：`P2/grouped_risk_output_compact`
- AURC 差值：-0.006407；候选胜出 3/5 折。
- 错误检测 AUROC 差值：-0.006213；候选胜出 1/5 折。
- 双主指标判定：**未同时通过**。

## 4. 结果解释

V3-3 没有同时通过两个主指标，但成功改善了它直接优化的连续风险目标。validation 选择器的聚合 AURC 从普通确定度的 0.345228 降到 0.340077，差值为 -0.005150；80% 覆盖率 masked BCE 降低 0.005135，Macro-F1 提高 0.004558。AURC 在 3/5 折胜出，Risk@80% 在 4/5 折胜出。

与此同时，selected 的错误检测 AUROC 从 0.727265 降到 0.707375，差值为 -0.019890，并且 0/5 折胜出。这符合目标差异：Ridge 直接学习连续 BCE，而 AUROC 的正例是“至少有一个标签错误”。一个样本可以出现低幅度的单标签错误但总体 BCE 不高，也可以没有阈值错误却有较高概率损失。

full-compact 相比 output-compact 将 AURC 降低 0.006407、Risk@80% 降低 0.002160，但错误检测 AUROC 下降 0.006213。传感器特征因此对连续风险排序有增量价值，但没有改善离散错误事件检测。validation 在 15 个单元中选择 full 11 次、output 4 次，全部四折用户重叠数为 0。

工程上不应强迫一个分数同时承担两个不同目标：如果系统目标是降低已接受样本的平均 BCE，应使用 V3-3 grouped-risk；如果目标是发现任一标签是否出错，P2 普通二元确定度仍然更合适。

## 5. 解释边界

所有 α 与特征家族选择均来自 validation 用户分组 OOF AURC；test 只用于最终评分。selected 曲线必须与 output/full 原始曲线一起解释。

V3-3 的 primary success 为 false，表示“双主指标同时改善”的严格假设未通过；这不应被改写成全面成功。连续风险改善属于预声明指标中的真实部分收益。

所有 α 与特征家族选择均来自 validation 用户分组 OOF AURC；test 只用于最终评分。selected 曲线必须与 output/full 原始曲线一起解释。

## 6. 可复核产物

每个单元均保存 α 候选、四折用户隔离合同、最终 Ridge 参数、曲线和摘要。
