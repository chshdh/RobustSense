# RobustSense V3-3 实验协议：用户分组连续风险选择器

## 1. 目标

在不重训 P2 的前提下，用与 AURC 直接一致的连续 BCE 风险目标、用户隔离交叉验证和低维特征，构建跨用户更稳定的选择性预测分数。

## 2. 矩阵与边界

- 主模型：P2。
- fold × seed：5 × 3，共 15 个单元。
- validation：风险模型拟合、α 选择和特征家族选择。
- test：最终一次性评分。
- 主模型 checkpoint、逐标签阈值和数据划分全部只读复用。
- 重新推理的 test 概率必须在 `1e-6` 内匹配冻结 Parquet。

## 3. 特征

output-compact 六维：标签确定度均值与最小值、阈值间隔均值与最小值、预测阳性比例、标签概率标准差。

full-compact 在六维基础上增加五维：可用模态比例、系统可靠度、可用模态可靠度均值与最小值、融合权重集中度。

用户 ID 只用于划分 GroupKFold，绝不作为模型输入。

## 4. 风险模型和选择规则

目标是每个样本在已知标签上的平均二元交叉熵。标准化统计量只能来自相应拟合子集。

每个特征家族使用四折用户分组交叉验证，对 α = 0.1、1、10、100、1000 产生完整 OOF 风险预测；将预测风险单调转换为置信度后，按 0.05 至 1.00 覆盖率计算 OOF AURC。选择最低 AURC 的 α，精确平局选择更大的 α。

随后比较 output/full 的最低 OOF AURC，选择 `grouped_risk_selected`；精确平局选择 output。最后用全部 validation 重拟合所选 α。

## 5. 正式报告选择器

1. `mean_binary_certainty`；
2. `p2_system_reliability`；
3. `grouped_risk_output_compact`；
4. `grouped_risk_full_compact`；
5. `grouped_risk_selected`。

selected 与某一原始曲线重复是预期行为，必须保留其 validation 选择身份。

## 6. 主比较

- selected 对普通分类确定度：检验分组连续风险方法的整体收益；
- full-compact 对 output-compact：检验压缩后的传感器特征是否仍有增量价值。

主指标仍为归一化 AURC 与错误检测 AUROC；辅助指标沿用固定覆盖率风险、Macro-F1、Micro-F1 和错误检测 AUPRC。

## 7. 禁止事项

- 禁止 test 参与 α、特征家族或模型选择；
- 禁止同一用户跨 GroupKFold 的拟合侧与验证侧；
- 禁止训练或修改 P2 checkpoint；
- 禁止只报告 selected 而隐藏 output/full 候选；
- 禁止在看到 test 后扩充 α 网格或特征集合。
