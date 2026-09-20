# RobustSense V2-4 三种子核心矩阵与扩展压力评估协议

- 协议 ID：V2-4
- 状态：已冻结；首个 V2-4 test 前锁定
- 父协议：V2-3
- 日期：2026-09-18

## 1. 核心矩阵

```text
models = [B4, B5, P, P2]
folds  = [0, 1, 2, 3, 4]
seeds  = [13, 29, 47]
units  = 4 × 5 × 3 = 60
```

B4、B5、P 分别映射到 V1 的 `gated`、`robust-gated`、`quality-aware`。seed 13 的
15 个 V1 单元和 5 个 P2 单元通过合同验证复用；seeds 29、47 新训练 40 个单元。

## 2. 训练与阈值合同

- 外层 fold 为 test，下一 fold 为 validation，其余为 train；
- batch 512、最大 80 epochs、AdamW、学习率 0.001、weight decay 0.0001；
- patience 10，早停指标为 validation Macro-F1@0.5；
- 逐标签阈值只在 validation 的 0.05–0.95 网格选择；
- P2 拒绝阈值只在 validation 选择，目标 coverage 为 90%；
- train-only 统计、未知标签 mask 和样本级确定性污染语义与父协议一致；
- test 只在 checkpoint、分类阈值和拒绝阈值冻结后由独立评估进程打开。

## 3. 扩展压力测试

每个 `model × fold × seed` 使用同一完整模态测试总体和相同有序样本 ID：

- 63 个非空 availability masks；
- 60 个 `drop one + Gaussian another` 混合故障；
- episode 长度 5、15、30，fault 为 drop 或 Gaussian sigma 2，时间断点 90 秒；
- 可靠度校准固定 10 bins；
- P2 的 Risk-Coverage 使用 validation 阈值和冻结 coverage 网格；
- 扩展场景只保存聚合指标、诊断、样本哈希和 lineage，不保存全量逐场景预测。
- 扩展推理把 8 个场景合并成一个 GPU 批次；场景语义和样本顺序不变，降低小模型的
  CPU 调度与主机到显存传输开销；若 6GB 显存不足则该单元失败，不在运行后静默改协议。
- 持续故障检测阈值固定为 0.5，恢复容差固定为 0.05，不读取 test 标签调参。

## 4. 统计顺序

1. 每个 `fold × model × scenario` 先对三个 seeds 聚合；
2. 再跨五个 fold 报均值和样本标准差；
3. P2 与 P 的主要比较使用同 fold、同 seed、同用户和同场景；
4. 自然缺失主要终点使用用户级配对 Bootstrap，2,000 次，seed 2404；
5. 保存抽样索引哈希、源 run ID、有效重复数和 2.5%/97.5% 区间。

## 5. 可选扩展消融

P2-A3、P2-A1 来自 V2-3 的 validation-only 冻结选择。可为 seeds 29、47 各运行五折，
共 20 个单元。它们不改变核心 60 单元的完整性判断，也不能因 test 表现被替换。

## 6. 验收

1. 60 个核心逻辑单元全部 success，其中 20 个复用合同可追溯、40 个新训练可恢复；
2. 所有扩展场景在 B4/B5/P/P2 间共享同一有序样本哈希；
3. 聚合顺序符合本协议，失败单元不得进入聚合；
4. Bootstrap 按用户配对且可复现；
5. test 不参与阈值、复用、场景或消融选择；
6. 协议锁、完整测试、依赖检查、V1 813 文件和 V1 报告重建全部通过。
