# Phase V2-6 最终验收报告

- 阶段：V2-6
- 状态：内部发布候选通过
- 日期：2026-09-19
- PPT：不在范围内
- 虚拟环境复现：不在范围内

## 1. 阶段结论

V2 最终技术材料、模型卡、实验协议总览、性能优化案例、两分钟与八分钟讲稿、30 题
面试问答、Demo 演示脚本、中文讲解动图、来源映射和发布审查均已完成。核心矩阵为
`60/60 success`，扩展压力评估为 `60/60 success`，Demo 2.0 真实四模型回放通过。

本阶段适合本地面试演示、导师汇报和私有仓库评审。公开发布仍被两个项目所有者决策
阻塞：选择代码许可证，以及在 `CITATION.cff` 中填写真实作者信息。这不影响内部验收，
但在两个事项完成前不得把仓库标记为可公开发布。

## 2. 最终材料

| 材料 | 用途 |
|---|---|
| `TECHNICAL_REPORT_V2.md` | 方法、实验、统计解释、限制与结论 |
| `MODEL_CARD_V2.md` | 适用范围、风险、指标和非预期用途 |
| `EXPERIMENT_PROTOCOL_V2_FINAL.md` | 数据切分、阈值、场景和聚合合同 |
| `PERFORMANCE_OPTIMIZATION_CASE.md` | CPU 瓶颈、双任务并行与场景合批案例 |
| `ORAL_SCRIPT_2MIN_V2.md` | 电梯式项目陈述 |
| `ORAL_SCRIPT_8MIN_V2.md` | 完整面试讲解结构 |
| `INTERVIEW_QA_V2.md` | 30 题追问、推荐回答与禁用表述 |
| `DEMO_RECORDING_SCRIPT.md` | 约 45 秒演示顺序和口播 |
| `demo_v2_walkthrough.gif` | 7 章、45 秒中文讲解动图 |
| `RELEASE_REVIEW_V2.md` | 数据、隐私、许可证与发布边界 |
| `source_map.csv` | 数值结论到权威产物和 run 集合的映射 |

全部新增说明文档使用中文。必要的模型名、指标名和代码标识保留英文，以便和实现、
论文术语及产物字段一一对应。

## 3. Demo 2.0 复核

通过本机页面实际检查了五个功能标签页：同样本对比、质量与门控、持续故障时间轴、
聚合背景和产物合同。页面加载 fold 0 / seed 29 的真实 P2 checkpoint，显示系统可靠度、
接受/拒绝状态、真实 run、测试样本和报告来源。页面不存在随机预测回退；合同不匹配会
停止运行。

讲解动图不是伪造单样本概率，而是从以下权威产物自动读取数值后生成：

- `phase_v2_4_core_summary.csv`；
- `phase_v2_4_natural_bootstrap.json`；
- `phase_v2_4_mask_summary.csv`；
- `phase_v2_4_mixed_summary.csv`；
- `phase_v2_4_persistent_summary.csv`；
- `phase_v2_5_acceptance.json`。

动图明确保留离线特征回放边界，也明确说明 B5 在自然缺失、63-mask 和持续故障分类
均值上更高，P2 的优势主要是混合故障描述性结果、诊断信息与选择性拒绝闭环。

## 4. 最终回归

```text
Ruff                         All checks passed
pip check                    No broken requirements found
pytest                       78 passed, 0 failed, 0 errors
V1 baseline manifest         813/813 files verified
V2-4 core                    60/60 success
V2-4 extended                60/60 success
V2-5 real replay             passed
Streamlit health             HTTP 200, ok
Demo GIF                     7 frames, 45 seconds
```

主 `README.md` 属于 V1 冻结范围。终验曾发现其被 V2 摘要改写，因此已把 V2 入口迁移到
`docs/v2/README.md`，并将主 README 精确恢复到冻结版本；最终 V1 校验为 813/813。

## 5. 科学结论

自然缺失 Macro-F1 均值为 B4 `0.565135`、B5 `0.578605`、P `0.576678`、P2
`0.577684`。P2 相对 P 的用户级配对 Bootstrap 均值为 `+0.001184`，95% 区间为
`[-0.001732, 0.004345]`，区间跨 0。不能声称 P2 在主终点上统计稳定地全面优于 P。

在扩展评估中，P2 的混合故障平均 Macro-F1 为 `0.565422`，四模型最高；B5 的
63-mask 平均 `0.517801` 和持续故障 fault 阶段平均 `0.594575` 最高。P2 的工程与面试
价值是把任务 utility 与输入 reliability 分开，提供系统可靠度和 validation 冻结的拒绝
决策，而不是把它包装成所有分类指标第一。

## 6. 最终边界

- 数据是 ExtraSensory 预计算特征，不是实时手机原始信号；
- 人工污染发生在标准化特征空间，不等同于物理传感器故障；
- 模型未验证生产级安全保证、跨数据集泛化或移动端实时部署；
- 用户 ID 哈希不等于安全匿名化，公开材料不得包含可反查个体的逐样本时间序列；
- 原始数据、处理数据和大 checkpoint 不应随公开仓库提交；
- 许可证与真实作者信息完成前，公开发布状态保持阻塞。

## 7. 阶段门

V2-6 内部阶段门通过。最终发布锁记录本阶段全部材料的字节数与 SHA-256，并验证父级
V2-5 协议锁、真实 Demo 验收、HTTP 健康检查、核心/扩展完成数、pytest 结果和 V1
冻结清单。后续若修改任一锁定材料，应生成新的版本化发布锁，不得覆盖本次证据。
