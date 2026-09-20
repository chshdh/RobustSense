# Phase 5 报告：正式实验已验收

## 结果

Phase 5 已针对冻结的五折、单种子 `credible` 配置完成并验收。全部注册实验单元成功结束，自动报告验证器接受所有必需产物。

- 冻结协议 SHA-256：`91a4e5fd7547ecf347d7fb3d7ee101d7389ac71f77386d4301e1bc1ec8929a7b`。
- A0–A5 开发单元：6/6 成功；失败/重试/跳过：0/0/0。
- 可信单元：60/60 成功；失败/重试/跳过：0/0/0。
- 官方外层 fold：0–4；每个 fold 都有仅由训练数据拟合的独立预处理器。
- 可信种子：`[13]`。
- 完整性状态：`passed`；验证 60 个 run ID，并为每个 fold 保存一个受控子集样本 ID 哈希。
- 创建协议锁后，没有修改配置、阈值规则、标签集、种子或场景。

事实来源为 `reports/run_registry.csv`、`reports/run_attempts.csv`、`reports/run_completeness.json` 和 `reports/phase5_protocol_lock.json`。

## 注册结果摘要

只有 60-run 矩阵通过完整性验证后，才重新生成最终技术报告和全部 E1–E8 表。

| 评估 | Macro-F1 最佳模型 | 均值 | 标准差 |
|---|---|---:|---:|
| 受控完整模态 | robust-gated | 0.6043 | 0.0253 |
| 自然缺失 | quality-aware | 0.5781 | 0.0292 |

Quality-Aware 在自然缺失 Macro-F1 上比普通 Gated Fusion 平均高 `+0.0110`，并在全部五个配对外层 fold 中获胜。Robust Gated 的受控完整结果最佳，但相对 Gated 的平均优势仅为 `+0.0048`，且只在五个 fold 中的三个获胜。这些是描述性五折结果；可信配置只有一个种子，无法估计种子级不确定性。

随机移除三个模态时，Gated、Robust Gated 和 Quality-Aware 的平均相对 Macro-F1 退化分别为 12.87%、10.33% 和 9.45%，支持鲁棒性目标。可靠度输出会随 Gaussian、bias 和 scale 污染明显下降，但目标模态融合权重在 Gaussian 噪声或 bias drift 下不会一致下降，因此门控可解释性假设仅得到部分支持。

以上数值由报告生成器从 `reports/tables/` 复现。`reports/source_map.csv` 记录产物到 run 的映射，`reports/readme_source_map.csv` 提供核心结论的逐声明筛选条件。

## A0–A5 开发筛查

消融筛查使用 fold 0、种子 13 和三 epoch 开发预算。它只作为工程证据保留，不提升为五折因果结论。

| ID | 改动 | 自然 Macro-F1 | 受控完整 Macro-F1 |
|---|---|---:|---:|
| A0 | 完整 P | 0.5582 | 0.5911 |
| A1 | 无 Sensor Dropout | 0.5442 | 0.5798 |
| A2 | 无噪声增强 | 0.5537 | 0.5922 |
| A3 | 无显式质量特征 | 0.5559 | 0.5983 |
| A4 | 无 ReliabilityNet，仅内容门控 | 0.5770 | 0.5923 |
| A5 | 无可靠度辅助损失 | 0.5531 | 0.5920 |

混合排序意味着不能声称每个组件都能独立提高干净和自然性能。查看筛查结果后没有修改可信配置。

## 执行治理

Sweep 运行器保证：

- 普通恢复只选择 `pending` 单元；
- 不无声重跑成功单元；
- 失败单元必须显式使用 `--retry-failed`；
- 每次尝试都保留命令、时间戳、返回码和日志路径；
- 可选 `full` 配置在可信矩阵达到 60/60 前保持锁定；
- 最终聚合器拒绝缺失 fold、种子、清单、输出或不一致的受控样本 ID 哈希。

全部 66 个注册单元都在首次记录的尝试中成功。详细日志位于 `runs/_phase5_logs/`。

## 生成证据

- `reports/technical_report.md`：中文摘要和完整注册分析。
- `reports/tables/`：E1–E8、逐标签、鲁棒性 AUC 和消融表。
- `reports/figures/`：完整必需图集。
- `reports/run_completeness.json`：60/60 可信验证结果。
- `reports/source_map.csv`：产物级来源追踪。
- `reports/readme_source_map.csv`：README 声明级来源追踪。

各 fold 的受控完整总体分别为 35,613、42,146、59,898、48,501、22,393 条。自然缺失使用每条合法测试记录。Gaussian、bias 和 scale 场景仍是标准化特征空间压力测试，不得描述为物理传感器故障模拟。

## 验证与下一道门

报告重建验证了全部 60 个可信 run，并生成 13 个表和 11 张正式可信图；单独的开发消融图保留其六个开发源 run。最终报告升级后的定向报告测试和 Ruff 均通过。

Phase 6 已解除阻塞。其 Streamlit 应用必须读取真实 run 的检查点、预处理器、阈值、模式和测试样本；不得使用随机或硬编码预测替代，并必须明确标记为离线回放而非实时传感器采集。
