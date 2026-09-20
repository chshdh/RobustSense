# RobustSense V3-2 实验协议：Validation 拟合的任务对齐置信度

## 1. 目标

检验轻量任务置信度层能否修复 V3-1 中“模态质量可靠度不等于分类正确置信度”的目标错位，并分离分类输出特征与传感器可靠度特征的贡献。

## 2. 实验矩阵

- 主模型：P2。
- fold：0、1、2、3、4。
- seed：13、29、47。
- 拟合单元：15 个；每单元两个 validation 拟合选择器。
- test 选择器曲线：每单元 4 条，共 60 条。
- 场景：自然缺失。
- 主模型 checkpoint、逐标签分类阈值、数据划分全部复用冻结 V2/V3-1 产物。

## 3. 数据边界

1. selector 只在 validation 上拟合。
2. test 标签只能在 selector 完全固定后用于评分。
3. validation 与 test 都只保留至少有一个已知目标标签的有效样本。
4. selector 输入特征全部可在推理时得到；真实标签只用于生成 validation 错误目标。
5. 重新前向推理的 test 分类概率必须与冻结 Parquet 在 `1e-6` 绝对误差内一致。

## 4. 选择器

基线：

- `mean_binary_certainty`；
- `p2_system_reliability`。

拟合选择器：

- `task_confidence_output_only`：每标签二元确定度与每标签验证阈值间隔，共 30 维；
- `task_confidence_full`：在 30 维基础上加入模态可用性、融合权重、掩码后的模态可靠度与系统可靠度，共 49 维。

所有连续特征只使用对应 validation 的均值和标准差进行标准化。零方差特征的 scale 固定为 1。逻辑回归超参数固定，不做搜索。

## 5. 指标与聚合

覆盖率、AURC、错误检测指标、固定覆盖率指标和聚合方法与 V3-1 完全一致：覆盖率 0.05 至 1.00、步长 0.05；先对同 fold 三个 seed 取均值，再对五 fold 等权平均。

主指标：

- 归一化 AURC，越低越好；
- 错误检测 AUROC，越高越好。

主比较：

1. full 对普通分类确定度：回答任务对齐层是否带来整体收益；
2. full 对 output-only：回答传感器可靠度相关特征是否有增量价值。

## 6. 禁止事项

- 禁止查看 test 结果后调整逻辑回归超参数或特征集合；
- 禁止按 test 结果挑选 output-only 或 full；
- 禁止使用 test 标签训练、校准或早停；
- 禁止把 validation 拟合层表述为重新训练了 P2 主模型。

## 7. 产物

- 协议锁：`reports/v3/phase_v3_2_protocol_lock.json`；
- 单元产物：`reports/v3/phase_v3_2/units/fold{fold}_seed{seed}/`；
- 最终摘要：`reports/v3/phase_v3_2_summary.csv`；
- 机器报告：`reports/v3/phase_v3_2_report.json`；
- 中文报告：`docs/v3/PHASE_V3_2_REPORT.md`。
