# Phase V2-2 验收报告：扩展评估套件

- 状态：通过
- 完成日期：2026-09-17
- 协议状态：开发实现已冻结
- 协议哈希：`8d3159316822bd368e8fa155887d277e7055c385457a6a403d7ff785e7561ff5`
- 父协议哈希：`558b0082dd40932847e04c30e9af3f237514a78ee38168abc7def5dcbae2e4ad`
- 边界：本阶段只完成评估工具和协议产物，未执行正式五折训练，也未生成模型成绩

## 1. 完成内容

### 场景与样本合同

- 枚举全部 63 个非空六模态 mask，位序与 `MODALITIES` 常量一致；
- 枚举 60 个有序 `drop × Gaussian × sigma` 混合故障，强制 drop/noisy 不同；
- 持续 episode 按用户、时间戳排序，在相邻间隔大于 90 秒处断开；
- 支持 5、15、30 步的 pre/fault/post 三段窗口，并记录短段排除数和未使用行数；
- 使用有序 `user_id|timestamp` 计算场景样本哈希；聚合前强制所有核心模型共享同一哈希；
- 场景规范清单为稳定 JSONL，相同输入重复生成字节一致。

### 指标与诊断

- Mask-only 模型的 forward 只接收六维 availability；
- 用户内 availability 确定性置换，不允许跨用户交换；
- 可靠度固定分箱校准、ECE，以及 `1-r_i` 的故障 AUROC/AUPRC；
- Risk-Coverage 曲线包含 masked BCE、Macro-F1、Micro-F1；
- 持续故障包含 pre/fault/post 分类指标、检测延迟和恢复到 pre 基线范围的步数；
- 63-mask 聚合包含宏平均、自然频率加权、最坏值及按可用模态数分层；
- 用户级配对 Bootstrap 默认 2,000 次，保存 seed、抽样索引哈希和两个源 run ID；
- 所有空 bin、单类指标、空阶段和无有效统计保留 `NaN`。

### 产物追溯

- 稳定 CSV 固定列顺序、浮点格式与 `NaN` 表示；
- artifact lineage 强制每个图表至少有一个 `source_run_id`；
- 当前生成的 scenario artifacts 明确标记为“场景合同，不是模型结果”。

## 2. 主要文件

| 类别 | 文件 |
|---|---|
| 场景实现 | `src/robustsense/evaluation/v2_scenarios.py` |
| 指标实现 | `src/robustsense/evaluation/v2_metrics.py` |
| 聚合与哈希闸门 | `src/robustsense/evaluation/v2_aggregate.py` |
| 稳定产物与 lineage | `src/robustsense/evaluation/v2_artifacts.py` |
| Mask-only | `src/robustsense/models/mask_only.py` |
| 场景构建脚本 | `scripts/build_v2_phase2_contract.py` |
| 协议锁脚本 | `scripts/lock_v2_phase2_protocol.py` |
| 决策记录 | `docs/v2/ADR-0003-扩展评估合同.md` |
| 冻结协议 | `docs/v2/EXPERIMENT_PROTOCOL_V2_2.md` |
| 协议锁 | `reports/v2/phase_v2_2_protocol_lock.json` |

## 3. 场景产物

| 产物 | 数量/用途 | SHA-256 |
|---|---:|---|
| `availability_masks.jsonl` | 63 masks | `cec55ec801baa8c385ee141a13038b2daa50f44c9456c4054f9e63f1f2841e2e` |
| `mixed_failures.jsonl` | 60 mixed faults | `8077e02d8747e4a733b29673f1aeb649b0ee926cc72eb40c00599f94adae1ac3` |
| `scenario_contract.json` | 场景总合同 | `fabb12ee05942802dcfbc2b4b06fabe2c0c4b1f6afd1aa0a8732aa27de03ed80` |

连续构建前后逐文件哈希完全相同，`ByteStable = true`。

## 4. 自动化验收

### 针对性测试

V2-2 新增 17 项测试，结果：`17 passed`。覆盖：

- 63-mask 完整性、唯一性和清单字节稳定性；
- 60 个混合故障及 drop/noisy 隔离；
- episode 用户和时间断点；
- 样本哈希稳定性与跨模型哈希漂移拒绝；
- Mask-only 输入边界和用户内置换；
- 校准空 bin、单类 AUROC、空 phase 的 `NaN`；
- Risk-Coverage 的未知标签屏蔽；
- 用户级配对 Bootstrap 可复现；
- CSV 字节稳定和图表 run lineage 强制检查。

### 全项目回归

```text
ruff check .                         All checks passed
pytest                              62 passed in 26.59s
pip check                           No broken requirements found
V1 baseline manifest               813/813 files verified
V1 report rebuild                  31 artifacts, 0 changes on both rebuilds
scenario contract repeated build   ByteStable = true
```

V1 重建证据保存于 `reports/v2/v1_report_rebuild_after_v2_2.json`。

## 5. 验收条件映射

| 规约要求 | 验收结果 |
|---|---|
| 所有模型共享同一场景样本哈希 | 已实现强制验证；不一致直接失败 |
| 重复生成字节稳定 | 三份场景产物复建哈希无变化 |
| 空指标保留 NaN | 单元测试覆盖空 bin、单类、空 phase 和 0 coverage |
| 图表可追溯到 run ID | lineage writer 拒绝无 `source_run_ids` 的产物 |
| Bootstrap 以用户为单位 | 已实现用户有放回、成对抽样和索引哈希 |
| V1 不被污染 | 813 个文件通过；31 个报告产物复建零变化 |

## 6. 科研边界与下一阶段

本报告中的数量和哈希只证明评估合同可复现，不是模型性能结果。尚未运行 B4/B5/P/P2 的
正式 63-mask、混合或持续故障测试，也没有产生可用于论文或面试陈述的性能比较。

下一阶段是 V2-3：冻结 seed 13 五折与正式消融协议，随后运行 P2 五折和 A1–A4 消融。
正式运行时必须复用本阶段的场景清单、validation 阈值、样本哈希闸门和 artifact lineage。
