# RobustSense V2 文档索引

- `ADR-0001-V1保护与V2隔离.md`：V1 冻结与目录隔离决策。
- `ADR-0002-P2门控与验证集拒绝.md`：P2 门控、ranking 视图与拒绝阈值决策。
- `ADR-0003-扩展评估合同.md`：63-mask、混合/持续故障、统计与产物追溯决策。
- `ADR-0004-seed13五折与消融治理.md`：seed 13 五折、A1–A4 消融和可恢复运行治理。
- `EXPERIMENT_PROTOCOL_DRAFT.md`：V2-0 协议边界与验收规则。
- `EXPERIMENT_PROTOCOL_V2_1.md`：P2 最小实现的已冻结开发协议。
- `EXPERIMENT_PROTOCOL_V2_2.md`：扩展评估套件的已冻结协议。
- `EXPERIMENT_PROTOCOL_V2_3.md`：seed 13 五折与正式消融的已冻结协议。
- `PHASE_V2_0_REPORT.md`：本阶段完成后由实测证据填写的验收报告。
- `PHASE_V2_1_REPORT.md`：P2 最小实现与回归验收报告。
- `PHASE_V2_2_REPORT.md`：扩展评估套件与回归验收报告。
- `PHASE_V2_3_REPORT.md`：25 个正式单元、五折结果、消融解释和最终验收报告。
- `PHASE_V2_4_REPORT.md`：60 个核心单元、60 个扩展压力评估单元、多种子统计与验收报告。
- `PHASE_V2_5_REPORT.md`：真实 checkpoint 驱动、失败关闭的 Demo 2.0 验收报告。
- `TECHNICAL_REPORT_V2.md`：V2 方法、正式结果、统计解释与限制。
- `MODEL_CARD_V2.md`：适用范围、风险、指标与非预期用途。
- `EXPERIMENT_PROTOCOL_V2_FINAL.md`：从数据切分到最终聚合的冻结实验协议总览。
- `PERFORMANCE_OPTIMIZATION_CASE.md`：CPU/GPU 瓶颈、并行调度和场景合批案例。
- `ORAL_SCRIPT_2MIN_V2.md`：两分钟面试口述稿。
- `ORAL_SCRIPT_8MIN_V2.md`：八分钟完整项目讲解稿。
- `INTERVIEW_QA_V2.md`：30 题面试追问、推荐回答和错误表述。
- `DEMO_RECORDING_SCRIPT.md`：约 45 秒 Demo 演示顺序与边界声明。
- `RELEASE_REVIEW_V2.md`：数据、隐私、许可证与公开发布审查。
- `PHASE_V2_6_REPORT.md`：V2 最终交付、回归与内部发布验收报告。

## V2-6 快速入口

- 中文讲解动图：`reports/v2/demo_v2_walkthrough.gif`；
- 数值来源映射：`reports/v2/source_map.csv`；
- 最终发布锁：`reports/v2/phase_v2_6_release_lock.json`；
- Demo 启动命令：`.\scripts\run_demo.ps1`。

主 `README.md` 属于 V1 冻结的 813 个文件之一，因此 V2 不再修改该文件。V2 的新增说明、
结果和面试材料以本索引及 `docs/v2/` 为准。
