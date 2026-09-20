# RobustSense V2-3 seed 13 五折与正式消融协议

- 协议 ID：V2-3
- 状态：首个正式 test 运行前冻结
- 父协议：V2-2
- 冻结日期：2026-09-17

## 1. 实验矩阵

```text
folds    = [0, 1, 2, 3, 4]
seeds    = [13]
variants = [P2, P2-A1, P2-A2, P2-A3, P2-A4]
units    = 5 × 1 × 5 = 25
```

外层 fold `f` 为 test，`(f+1) mod 5` 为 validation，其余三个 fold 为 train。沿用现有
fold-specific 中位数/IQR、类别权重、15 标签顺序和未知标签 mask。

## 2. 训练合同

- 最大 epoch：80；早停 patience：10；早停指标：validation Macro-F1@0.5；
- batch：512；优化器：AdamW；学习率：0.001；weight decay：0.0001；
- 梯度裁剪：5.0；pos weight 上限：20；
- 逐标签阈值网格：0.05–0.95，步长 0.05，只在 validation 选择；
- 目标 coverage：90%，拒绝阈值只在 validation 选择；
- 训练污染定义、概率和样本级 RNG 与 V2-1 完全一致；
- 正式代码只使用经过逐元素等价测试的批量污染写入实现。

训练阶段禁止构造 test dataset。训练产物状态必须先达到
`trained_validation_frozen_test_not_opened`，之后才允许独立 test 评估进程运行一次。

## 3. 消融定义

| ID | 唯一改动 | 其余部分 |
|---|---|---|
| P2 | 完整方法 | 全部启用 |
| P2-A1 | 权重只由 utility masked softmax 得到 | ReliabilityNet、辅助损失和选择性预测保留 |
| P2-A2 | ranking loss 权重固定为 0 | 架构、其他损失和选择性预测保留 |
| P2-A3 | ReliabilityNet 只读取编码，不读取五维质量 | 可靠度约束、损失和选择性预测保留 |
| P2-A4 | 不设置拒绝阈值，只报告强制预测 | 参数由匹配 P2 checkpoint 无损派生 |

A4 的源和目标参数张量 SHA-256 必须相同；其 run 目录不得出现
`abstention_threshold.json` 或 `risk_coverage.json`。

## 4. 正式 test 输出

本阶段只评估自然缺失：

- 强制预测 Macro-F1、Micro-F1、mAP、Brier 和逐标签指标；
- P2/A1/A2/A3 同时报告 validation 阈值下的测试 coverage、接受子集指标和
  Risk-Coverage；
- A4 只报告强制预测；
- 每个结果记录 run ID、fold、seed、协议哈希、checkpoint 哈希、预处理清单哈希和阈值
  产物哈希。

本阶段不运行 63-mask、混合故障、持续故障或多种子统计。

## 5. 运行治理

- 权威注册表：`reports/v2/phase_v2_3_run_registry.csv`；
- 只追加尝试表：`reports/v2/phase_v2_3_run_attempts.csv`；
- 完整日志：`runs/v2/_phase_v2_3_logs/`；
- 正式 run：`runs/v2/phase_v2_3/`；
- pending 正常运行；failed 只能通过显式 retry；success 不得自动重跑；
- 中断时 running 转为 failed 并保留原因；
- 只有 25 个单元全部 success 才生成中期聚合。

## 6. V2-4 选择规则

是否进入 V2-4 的硬条件是 25 个单元完整、训练数值有限、合同全部通过；不要求 P2 必须
优于 P 或消融。若扩展消融种子，最多选择两个 A1–A4 变体：按五折 validation 中每个
变体相对 P2 的配对 Macro-F1 绝对差均值降序排列，同分按 ID 升序。test 不参与选择。

## 7. 阶段验收

1. 首个 test 前协议锁有效，V1 813 文件和 V2-2 父锁有效；
2. 25 个注册单元都有终态，成功必须为 25；失败和重试记录不得删除；
3. 每个 run 的 train/val/test 数据流、阈值来源和 checkpoint 合同可审计；
4. A4 与匹配 P2 参数哈希相同，且无选择性产物；
5. 中期表的所有行可追溯到 run ID；
6. V2-4 选择文件明确写出 `source_split=val` 和 `test_metrics_read=false`；
7. 完整测试、代码检查、依赖检查和 V1 重建零变化。
