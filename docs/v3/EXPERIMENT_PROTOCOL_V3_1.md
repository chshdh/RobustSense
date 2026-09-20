# RobustSense V3-1 实验协议：选择性预测公平评估

## 1. 实验目标

V3-1 不再追求微小的强制分类分数提升，而是检验系统能否在主动拒绝低置信样本时降低剩余样本风险，并检验 P2 的系统可靠度是否真正具有错误排序能力。

## 2. 冻结输入

- 模型：B4、B5、P、P2。
- 折次：0、1、2、3、4。
- 随机种子：13、29、47。
- 正式实验单元：`4 × 5 × 3 = 60` 个已有模型结果；选择器曲线共 `15 × 9 = 135` 条。
- 场景：自然缺失。
- checkpoint：只读复用 V2-4 成功登记的 checkpoint。
- 分类阈值：只读复用各 run 的验证集逐标签 F1 阈值。
- 测试总体：模型有效且至少含一个已知目标标签的样本。

## 3. 预声明选择器

每个模型都报告：

1. `mean_binary_certainty`：所有标签的平均二元确定度，主通用基线；
2. `mean_normalized_threshold_margin`：相对验证集分类阈值的平均归一化间隔，辅助通用基线。

P2 额外报告：

3. `p2_system_reliability`：模型输出的系统可靠度，候选方案。

因此每个 fold/seed 有 `4 × 2 + 1 = 9` 条选择器曲线。禁止增加测试标签派生的 oracle 排序，禁止在测试结果出现后删除表现不佳的预声明曲线。

## 4. 覆盖率与指标

覆盖率从 0.05 到 1.00，步长 0.05。按照置信度从高到低稳定排序，接受前 `ceil(coverage × N)` 个样本。

主指标：

- `normalized_aurc`：0.05 到 1.00 区间的 masked BCE 风险—覆盖率归一化面积，越低越好；
- `error_detection_auroc`：以 `1 - confidence` 检测样本是否至少有一个已知标签预测错误，越高越好。

辅助指标：

- 错误检测 AUPRC；
- 覆盖率 0.80、0.90、0.95、1.00 下的 masked BCE；
- 同覆盖率下的 Macro-F1 与 Micro-F1；
- 置信度均值、标准差、最小值、最大值以及错误流行率。

## 5. 公平性与泄漏控制

1. 所有选择器只使用推理时可得信息，不读取测试标签。
2. 真实标签只用于曲线完成后的风险和错误检测评分。
3. 四模型在每个 fold/seed 必须共享样本键、目标和已知标签掩码。
4. P2 重新前向推理得到系统可靠度后，其预测概率必须与冻结 Parquet 在 `1e-6` 绝对误差内一致。
5. 不允许根据测试结果修改置信度公式、覆盖率网格、主指标或主比较。
6. V3-1 不进行超参数调优、温度缩放或 checkpoint 选择。

## 6. 聚合规则

先在同一个 fold 内对三个 seed 取算术平均，再对五个 fold 等权取算术平均，并报告五个 fold 的样本标准差。主比较同时报告五个配对 fold 差值及胜出 fold 数。

## 7. 判读边界

- P2 系统可靠度优于 P2 普通确定度，才能说明可靠度选择器本身带来增益。
- P2 系统可靠度优于 B5 普通确定度，说明完整 P2 系统相对强基线具有风险控制优势，但该比较同时包含分类器与选择器差异。
- 即使选择性风险改善，也不能表述为所有样本上的分类准确率提高。
- 单一覆盖率上的偶然优势不能替代 AURC 和五折聚合结论。

## 8. 产物合同

- 协议锁：`reports/v3/phase_v3_1_protocol_lock.json`。
- 单元曲线与摘要：`reports/v3/phase_v3_1/units/fold{fold}_seed{seed}/`。
- 完整 fold/seed 曲线：`reports/v3/phase_v3_1_fold_seed_curves.csv`。
- 完整 fold/seed 摘要：`reports/v3/phase_v3_1_fold_seed_summary.csv`。
- fold 聚合与最终聚合：`reports/v3/phase_v3_1_fold_summary.csv`、`reports/v3/phase_v3_1_summary.csv`。
- 机器可读结论：`reports/v3/phase_v3_1_report.json`。
- 中文阶段报告：`docs/v3/PHASE_V3_1_REPORT.md`。
