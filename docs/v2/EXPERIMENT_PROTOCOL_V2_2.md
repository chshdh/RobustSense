# RobustSense V2-2 扩展评估协议

- 协议 ID：V2-2
- 状态：开发实现冻结
- 父协议：V2-1
- 本阶段不产生正式可信模型结论

## 1. 输入边界

- 模型：B4、B5、P、P2；Mask-only 仅作为捷径诊断。
- 正式样本：每个 fold 的完整模态测试总体。
- 分类阈值和拒绝阈值：只读相应 run 中由 validation 保存的产物。
- 样本身份：有序 `user_id|timestamp`，计算 SHA-256；同一 fold/seed/scenario 的所有核心
  模型必须相同。
- 禁止：按模型重采样、按测试结果调阈值、把开发 fixture 当作正式实验结果。

## 2. 场景矩阵

### 2.1 63-mask

按冻结模态顺序枚举整数 1–63 的位向量，产生全部 63 个非空 availability mask。
每个 mask 都作用于相同完整测试总体。报告 63-mask 宏平均、自然 mask 频率加权值、最坏
值，以及每种可用模态数量下的平均值和最坏值。

### 2.2 混合故障

有序枚举 `drop_modality != noisy_modality`，Gaussian sigma 取 1.0、2.0，共
`6 × 5 × 2 = 60` 个场景。先执行 drop，再执行 Gaussian；随机噪声由场景 ID、batch 和
seed 确定。

### 2.3 持续故障

按用户、时间戳稳定排序；相邻间隔大于 90 秒则切段。每个 episode 由等长 pre/fault/post
组成，长度分别取 5、15、30，故障类型为 drop、Gaussian sigma 2、bias 1。短段数量、
未使用行数和原因必须保存。报告三阶段分类指标、检测延迟和回到 pre 基线容差范围所需
步数。

## 3. Mask-only 诊断

Mask-only 只接收 `[N,6]` availability，不接收 features、feature masks 或质量特征。
按现有用户级五折训练，并在每个测试用户内部确定性置换 availability。原始与置换结果
分开报告，且明确说明它不是与完整传感器模型的公平性能竞赛。

## 4. 指标

- 分类：Macro-F1、Micro-F1、mAP、Brier 和每标签指标，始终尊重 target mask。
- 可靠度：固定 10-bin 校准表及 ECE；`1-r_i` 的故障 AUROC/AUPRC。
- 选择性预测：coverage 0.0–1.0，步长 0.1；接受样本 masked BCE、Macro-F1、
  Micro-F1，并单独读取 validation 冻结的 90% coverage 阈值结果。
- 统计：用户级配对 Bootstrap，默认 2,000 次，seed 13，2.5%/97.5% 分位区间。
- 空指标：必须为 IEEE `NaN`；序列化为字符串 `NaN`，禁止静默替换成 0。

## 5. 产物合同

- `reports/v2/scenarios/availability_masks.jsonl`：63-mask 规范清单；
- `reports/v2/scenarios/mixed_failures.jsonl`：60 个混合故障清单；
- `reports/v2/scenarios/scenario_contract.json`：场景数量、哈希和持续故障规则；
- 正式运行时每张结果表必须有 `run_id`；每个图必须在 lineage 中列出
  `source_run_ids`；
- 同一输入、配置和 seed 重复生成，场景清单必须字节一致。

当前三份 scenario 文件只描述协议，不含模型预测或正式成绩。

## 6. 验收

1. 63 个非空 mask 完整且唯一；60 个混合故障完整且 drop/noisy 不同。
2. episode 不跨用户或 90 秒断点，并报告短段与未使用行。
3. Mask-only 的函数签名无法接收传感器特征；用户内置换可复现。
4. 所有模型共享同一场景样本哈希；哈希漂移会使聚合失败。
5. 校准空 bin、单类 AUROC、空 phase 和无有效统计保留 `NaN`。
6. Bootstrap 以用户为抽样单位、成对计算且抽样索引哈希可复现。
7. 稳定 CSV、JSONL 重复生成字节一致；无源 run ID 的图表 lineage 被拒绝。
8. V1 报告重建零变化；V2-0、V2-1 协议锁保持有效；完整测试集通过。
