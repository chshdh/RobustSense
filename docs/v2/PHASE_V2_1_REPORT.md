# Phase V2-1 验收报告：P2 最小实现

- 状态：通过
- 完成日期：2026-09-17
- 协议状态：开发实现已冻结
- 协议哈希：`558b0082dd40932847e04c30e9af3f237514a78ee38168abc7def5dcbae2e4ad`
- 未执行：正式五折训练、test 调参、63-mask、混合/持续故障和正式结果声明

## 1. 本阶段完成内容

### P2 模型

新增 `ReliabilityConstrainedFusionModel`：

- 六个 V1 模态编码器保持不变；
- ReliabilityNet 读取编码与五维质量特征，输出 `[0,1]` 可靠度；
- UtilityNet 只读取编码，输出无界任务效用；
- `beta = softplus(raw_beta) + epsilon`，保证始终为正；
- 最终分数固定为 `utility + beta * log(reliability + epsilon)`；
- 不可用模态权重精确为 0，可用权重和为 1；
- 全模态不可用时直接报错；
- 输出 logits、weight、reliability、utility、system reliability 和可选 abstain。

### 损失与拒绝机制

- 新增 reliability ranking hinge loss；
- clean 与人工污染视图按同一样本、同一模态配对；
- ranking 只读取监督侧 `artificial_fault` mask，forward 不接收故障元数据；
- 拒绝阈值只能由 validation 系统可靠度选择；
- 目标覆盖率默认为 90%，ties 采用保守多接受策略；
- P2 开发训练完成后不自动读取 test，也不生成 `test_metrics.json`。

### 训练与加载合同

- P2 配置位于 `configs/v2/`，run 根目录为 `runs/v2/`；
- P2 checkpoint 合同版本为 2，V1 合同仍为版本 1；
- resolved config 保存 beta 初值、epsilon、损失权重、ranking margin、学习后 beta 和拒绝阈值产物；
- V1 与 V2 嵌套 run 路径通过 processed manifest 反向定位项目根，不再假设固定目录层数；
- 通用预测收集器可选收集 utility、system reliability 和 abstain，不改变 V1 已有字段。

## 2. 主要改动文件

| 类型 | 文件 |
|---|---|
| 模型 | `src/robustsense/models/fusion.py` |
| 损失 | `src/robustsense/training/phase3_losses.py` |
| 拒绝阈值 | `src/robustsense/training/selective.py` |
| 训练接入 | `src/robustsense/training/phase3_trainer.py` |
| 通用收集器 | `src/robustsense/training/trainer.py` |
| Pipeline/加载 | `src/robustsense/pipeline.py`、`src/robustsense/evaluation/suite.py` |
| P2 配置 | `configs/v2/models/reliability_constrained.yaml` |
| 开发污染 | `configs/v2/corruption/train.yaml` |
| 开发 profile | `configs/v2/evaluation/dev.yaml` |
| 单元测试 | `tests/unit/test_v2_p2.py` |
| 小型真实 fixture | `tests/integration/test_phase1_audit.py` |
| ADR/协议 | `docs/v2/ADR-0002-P2门控与验证集拒绝.md`、`docs/v2/EXPERIMENT_PROTOCOL_V2_1.md` |

## 3. 设计选择

1. 没有让 weight 直接等于 reliability。Utility 表示任务贡献，reliability 表示当前可信程度，
   约束公式让两者作用不同且可解释。
2. 没有用故障类型或强度作为推理特征。它们只生成监督 target 和 ranking mask。
3. ranking 使用 clean 对训练污染视图，不额外引入测试场景；clean reliability 保留梯度，
   consistency 仍 detach clean logits。
4. 拒绝阈值是独立 JSON 产物，明确记录 `source_split=val`、目标覆盖率、实际覆盖率和样本数。
5. P2 仍复用 V1 训练器的分类阈值、类别权重、用户划分和 unknown-label mask 语义。

## 4. 实际验证命令

```powershell
python -m pytest -q tests/unit/test_v2_p2.py
python -m pytest -q tests/integration/test_phase1_audit.py::test_small_real_shape_fixture_prepares_and_trains_b2_b4_p_and_p2
ruff check .
python -m pytest -q
python scripts/verify_v1_report_rebuild.py --project-root . --output reports/v2/v1_report_rebuild_after_v2_1.json
python scripts/freeze_v1_baseline.py verify --project-root . --manifest reports/v2/v1_baseline_manifest.json
python -m pip check
python scripts/lock_v2_phase1_protocol.py --project-root .
```

实际执行使用既有 `.venv-dev` 的 Python，并为 pytest 指定项目内短临时路径，以避开 Windows
沙箱默认临时目录权限和长路径限制。

## 5. 测试与回归结果

| 检查 | 结果 |
|---|---|
| P2 数学/接口单元测试 | 6 passed |
| 小型真实形状 P2 集成 fixture | 1 passed in 23.84s |
| 完整测试 | 45 passed in 24.54s |
| Ruff | All checks passed |
| `pip check` | No broken requirements found |
| V1 报告第一次重建 | 31 个产物，变化 0 |
| V1 报告第二次重建 | 31 个产物，变化 0 |
| V1 冻结清单 | 813/813 一致 |

集成 fixture 真实执行了数据审计、按训练统计预处理、P2 一轮训练、checkpoint 保存/重载、
validation 分类阈值和拒绝阈值生成。该 run 位于 pytest 临时目录，属于工程 fixture，不是研究
实验，也没有保留或读取 test 指标。

## 6. 验收条款映射

| 验收条款 | 证据 | 结果 |
|---|---|---|
| 不可用权重精确为 0 | `test_p2_output_contract...` | 通过 |
| 可用权重和为 1 | 同上 | 通过 |
| reliability 降低时固定 utility 权重不增加 | `test_reliability_constraint_is_monotonic...` | 通过 |
| `beta > 0` | softplus 参数化与单元断言 | 通过 |
| ranking loss 手算一致 | 手算期望 0.1 | 通过 |
| system reliability 手算一致 | `0.25×0.8+0.75×0.4=0.5` | 通过 |
| 推理删除故障元数据仍运行 | P2 batch 不含任何故障字段 | 通过 |
| checkpoint 重载输出一致 | logits/r/u/w/R/abstain 逐张量一致 | 通过 |
| 拒绝阈值只来自 validation | 非 `val` 调用抛错，产物记录来源 | 通过 |
| 合成与小型真实 fixture | 合成 batch 单测 + 真实形状集成测试 | 通过 |

## 7. 产物与边界

- V2-1 协议锁：`reports/v2/phase_v2_1_protocol_lock.json`。
- V1 回归证据：`reports/v2/v1_report_rebuild_after_v2_1.json`。
- 没有创建正式 P2 run，没有新的可信 test 数字，也没有改写任何 V1 run 或报告。
- 当前配置仍是开发起点，不能宣传 P2 已优于 P。

## 8. 下一阶段前置条件

Phase V2-2 可以开始扩展评估套件：63 个非空 mask、混合故障、持续故障、Mask-only、
reliability calibration、risk-coverage 和用户级 Bootstrap。开始前应先为场景样本 ID 哈希、
字节稳定性和 NaN 保留规则建立测试。CPU 污染向量化仍是性能优化候选，但不能改变已锁定的
样本级确定性和故障语义。

