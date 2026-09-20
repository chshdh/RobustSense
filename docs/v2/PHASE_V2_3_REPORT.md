# Phase V2-3 验收报告：seed 13 五折与正式消融

- 状态：通过
- 完成日期：2026-09-18
- 协议状态：首个正式 test 运行前冻结，完成后复核有效
- 协议哈希：`2d0aa3040711373d5db25ce22a7ea07812e03b218356545ce54c35173ae1ea90`
- 父协议哈希：`8d3159316822bd368e8fa155887d277e7055c385457a6a403d7ff785e7561ff5`
- 正式矩阵：5 folds × 1 seed × 5 variants = 25 units

## 1. 阶段结论

V2-3 的 25 个正式实验单元全部成功，最终注册表状态为 `success=25`、`failed=0`、
`pending=0`、`running=0`。训练阶段不构造 test dataset；早停、逐标签分类阈值和拒绝阈值
全部来自 validation，之后由独立进程打开 test 一次并写入评估产物。

P2 在 seed 13 五折上的强制预测 Macro-F1 为 `0.581311 ± 0.021659`。三个训练型消融
A1、A2、A3 的均值均略低于 P2；A4 只移除推理时拒绝机制，与对应 P2 共享完全相同的
参数和强制预测结果。当前结果支持“完整方法在该单种子五折矩阵中没有被任一训练型消融
稳定超过”，但不构成多种子统计显著性结论。

## 2. 五折结果

下表中的 `±` 为五个外层 fold 的样本标准差。

| 变体 | Validation Macro-F1 | Test 强制 Macro-F1 | Test Micro-F1 | Test mAP | Test coverage | 接受子集 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| P2 | 0.596940 ± 0.027913 | 0.581311 ± 0.021659 | 0.698617 ± 0.011124 | 0.580162 ± 0.024032 | 0.879375 ± 0.059567 | 0.581544 ± 0.021819 |
| P2-A1 | 0.595265 ± 0.026448 | 0.579168 ± 0.022344 | 0.696838 ± 0.019469 | 0.577820 ± 0.023847 | 0.892804 ± 0.073298 | 0.584434 ± 0.019966 |
| P2-A2 | 0.595713 ± 0.028212 | 0.580331 ± 0.024333 | 0.699936 ± 0.010746 | 0.577649 ± 0.026277 | 0.878543 ± 0.060569 | 0.581421 ± 0.024649 |
| P2-A3 | 0.593630 ± 0.025344 | 0.579808 ± 0.021674 | 0.697732 ± 0.017932 | 0.579572 ± 0.020761 | 0.899016 ± 0.020056 | 0.582798 ± 0.022784 |
| P2-A4 | 0.596940 ± 0.027913 | 0.581311 ± 0.021659 | 0.698617 ± 0.011124 | 0.580162 ± 0.024032 | 不适用 | 不适用 |

### 结果解释边界

- P2 相对 A1、A2、A3 的强制 Macro-F1 均值分别高 `0.002143`、`0.000980`、
  `0.001503`；差异较小，不能在未做多种子配对统计前声称显著优越。
- 90% 是 validation 上的目标 coverage，不保证 test 恰好为 90%。P2 的 test 平均 coverage
  为 87.94%，反映了 validation 到 test 的用户分布差异。
- 接受子集指标必须与 coverage 同时报告，不能只报告较高的选择性 Macro-F1。
- A4 与 P2 的强制指标相同是设计结果，不是独立训练带来的复现巧合。

## 3. 消融含义

| 变体 | 唯一改动 | 五折观察 |
|---|---|---|
| P2-A1 | 权重只由 utility masked softmax 得到 | 强制 Macro-F1 比 P2 低 0.002143；选择性分数较高但 coverage 不同 |
| P2-A2 | ranking loss 权重固定为 0 | 与 P2 最接近，强制 Macro-F1 低 0.000980 |
| P2-A3 | ReliabilityNet 不读取五维质量 | validation 配对影响最大，强制 Macro-F1 低 0.001503 |
| P2-A4 | 不做选择性拒绝 | 参数和强制预测与 P2 完全一致，不产生选择性产物 |

## 4. V2-4 验证集选择

冻结规则按五折 validation 中“相对 P2 的配对 Macro-F1 绝对差均值”降序选择最多两个
消融。结果为：

1. P2-A3：平均绝对配对差 `0.005236`；平均有符号差 `-0.003310`；
2. P2-A1：平均绝对配对差 `0.003459`；平均有符号差 `-0.001675`。

选择文件明确记录 `source_split=val`、`test_metrics_read=false`。这里选择的是对组件移除最敏感、
最值得在 V2-4 扩展种子的消融，不是按 test 成绩挑选“最好模型”。

## 5. 可复现性与治理审计

- 协议锁包含 21 个文件，完成后逐文件 SHA-256 复核全部通过；
- 25 个 run 均包含 checkpoint、解析后配置、validation 指标、阈值、test 指标、预测文件和
  评估清单；
- 5 个 A4 checkpoint 的 `state_dict_sha256` 均与同 fold、同 seed 的 P2 完全一致；
- 所有 A4 run 均不存在 `abstention_threshold.json` 和 `risk_coverage.json`；
- 两次运行环境中断分别发生在 P2-A1/fold1 和 P2-A2/fold3，首次尝试日志均保留；两个
  单元都按 `--retry-failed` 显式重试并成功，最终 `attempt_count=2`；
- 正式结果表 25 行，SHA-256 为
  `d5ef1a8ad9d5464e4be2c3d57276677585a2f6edbd2f6feb669e44d30a7509bc`；
- 五折汇总表 5 行，SHA-256 为
  `1c0bc40a7445a35af9cb8aaa3943160e1efb08462ec3492ddbecdff3b528541e`。

## 6. 最终自动化验收

```text
ruff check .                         All checks passed
pytest                              71 passed in 34.15s
pip check                           No broken requirements found
V2-3 registry                      25/25 success
V2-3 protocol lock                 21/21 files verified
P2-A4 state hash audit             5/5 exact matches
V1 baseline manifest               813/813 files verified
V1 report rebuild                  31 artifacts, 0 changes on both rebuilds
```

pytest 在当前 Windows 沙箱中使用项目内 `--basetemp` 并关闭 cache provider；此前两次执行的
错误均来自默认临时目录权限，不是测试断言失败。V1 重建证据保存于
`reports/v2/v1_report_rebuild_after_v2_3.json`。

## 7. 权威产物

| 产物 | 路径 |
|---|---|
| 冻结协议 | `docs/v2/EXPERIMENT_PROTOCOL_V2_3.md` |
| 协议锁 | `reports/v2/phase_v2_3_protocol_lock.json` |
| 正式注册表 | `reports/v2/phase_v2_3_run_registry.csv` |
| 尝试表 | `reports/v2/phase_v2_3_run_attempts.csv` |
| 25 行结果表 | `reports/v2/phase_v2_3_results.csv` |
| 五折汇总表 | `reports/v2/phase_v2_3_summary.csv` |
| 中期聚合 | `reports/v2/phase_v2_3_interim_report.json` |
| V2-4 选择 | `reports/v2/phase_v2_3_selection_for_v2_4.json` |
| 正式 run | `runs/v2/phase_v2_3/` |
| 完整运行日志 | `runs/v2/_phase_v2_3_logs/` |

## 8. 下一阶段

V2-3 到此结束。V2-4 应先冻结多种子与扩展评估矩阵，再执行：B4、B5、P、P2 的
seeds 13、29、47 核心矩阵，其中通过合同校验复用已有 seed 13，并补跑 seeds 29、47；
按 validation 规则选出的 P2-A3、P2-A1 只作为可选扩展消融。随后才进入 63-mask、混合
故障、持续故障和用户级配对 Bootstrap。不得根据本阶段 test 表重新选择消融或修改阈值。
