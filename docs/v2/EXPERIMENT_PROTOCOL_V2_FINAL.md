# RobustSense V2 最终实验协议

## 1. 冻结对象

- 数据：ExtraSensory 官方预计算特征与官方五折用户清单；
- 标签：15 个多标签目标，NaN 作为未知 mask；
- 模态：phone_acc、phone_gyro、watch_acc、location、audio、phone_state；
- 模型：B4、B5、P、P2；
- folds：0–4；seeds：13、29、47；
- 主指标：Macro-F1；辅助指标：Micro-F1、mAP、Brier、Masked BCE；
- 自然缺失主比较：P2-P 用户级配对 Bootstrap。

## 2. 划分与泄漏防护

外层 fold 为 test，下一 fold 为 validation，其余 fold 为 train。所有预处理统计、类别权重、
逐标签分类阈值和拒绝阈值只从 train/validation 获得。训练进程不构造 test dataset；独立
评估进程在 checkpoint 与阈值冻结后打开 test。

## 3. 模型与训练

统一 batch 512、最多 80 epochs、AdamW、学习率 0.001、weight decay 0.0001、patience 10。
B5/P/P2 使用 Sensor Dropout 和训练期特征污染。P2 使用分类、可靠度、排序与一致性损失。

## 4. 正式矩阵

核心矩阵 `4 models × 5 folds × 3 seeds = 60`。seed 13 的 20 个单元通过冻结合同复用，
seed 29/47 新训练 40 个。失败单元不得进入聚合；重试必须保留原始尝试记录。

## 5. 扩展场景

每个核心 run 在相同有序完整模态样本上执行 63-mask、60 个混合故障和 12 个持续故障
定义。持续故障长度为 5/15/30，时间断点 90 秒，检测阈值 0.5，恢复容差 0.05。

## 6. 聚合与统计

先对同 `model × fold × scenario` 的三个 seed 聚合，再跨五 fold 报均值与样本标准差。
自然缺失 P2-P 按外层 fold 分层，以用户为抽样单位进行 2,000 次配对 Bootstrap，seed 2404。
保存抽样索引哈希、有效重复数、源 run ID 和 2.5%/97.5% 分位数。

## 7. 结论规则

- 点估计最高不自动等同显著优势；
- Bootstrap 区间跨 0 时不得宣称稳定提升；
- 受控场景结论必须标记为特征空间实验；
- 不因 test 表现修改阈值、场景、消融选择或训练超参数；
- 不把 Demo 单样本现象替代正式聚合统计。

## 8. 权威锁

V2-0 至 V2-5 均有独立协议或执行修正锁。V2-4 核心协议哈希为
`35035f18ccb129c1f3b5d8b70f724d9059bf4f5d433259ba586b477d30bd9d34`；V2-5 Demo
协议哈希为 `d3d2a7254304686f8f67e234c92c8e49d44035f365ce7a87df593565c9b0b09f`。
