# RobustSense Phase V2-1 协议

## 范围

本协议只冻结 P2 最小实现及开发验证，不批准五折可信训练，不批准基于 test 的模型选择。

## P2 结构

- 编码器：与 V1 P 相同的六个模态专属编码器，默认输出 64 维。
- ReliabilityNet：`Linear(64+5,32) → GELU → Linear(32,1) → Sigmoid`。
- UtilityNet：`Linear(64,32) → GELU → Linear(32,1)`。
- `beta = softplus(raw_beta) + epsilon`，开发初值 1.0，`epsilon=1e-6`。
- `score_i = utility_i + beta * log(reliability_i + epsilon)`。
- 权重只在可用模态上 softmax；不可用权重为 0，可用权重和为 1。
- 系统可靠度为权重与模态可靠度的加权和。

## 开发损失

总损失为 masked weighted BCE、可靠度 MSE、一致性 MSE 和 ranking hinge loss 的加权和。
开发权重分别为 1、0.1、0.1、0.05，ranking margin 为 0.1。ranking 只比较同一样本、
同一模态的 clean 与人工污染视图，并只在人工故障 mask 上求平均。

## 阈值与数据隔离

- 分类阈值继续只从 validation 逐标签选择。
- 拒绝阈值只从 validation 的系统可靠度选择，开发目标覆盖率为 90%。
- checkpoint 不携带人工故障元数据；推理输入仍是 V1 batch 合同。
- P2 开发训练默认写入 `runs/v2/`，训练完成后不自动读取 test。
- 当前 test 评估、63-mask、risk-coverage 曲线和正式统计均属于后续阶段。

## 验收测试

单元测试覆盖 shape/顺序、可靠度范围、beta 正性、权重不变量、固定 utility 下的单调性、
ranking 手算、系统可靠度手算、validation-only 阈值、无故障元数据推理和 checkpoint 重载。
集成测试使用小型真实形状 fixture 完成 P2 train/validation，并断言没有 test 指标产物。

