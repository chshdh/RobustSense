# RobustSense

RobustSense 研究手机、手表、位置、音频和设备状态等模态在自然缺失或人工污染时的多标签上下文感知。项目使用 ExtraSensory 预计算特征，采用用户独立五折评估，并实现质量感知门控融合。

## 当前状态

**项目已完成从 V1 到 V4-1 的全部预定实验，并依据停止规则结束模型迭代。** V1 完成冻结的
五折 60-run `credible` 矩阵；V2 完成 60 个核心单元和 60 个扩展压力评估单元；V3 完成
四阶段选择性预测与双策略决策实验；V4-1 完成最后一次 5 折 × 3 种子的标签级融合优化。

最终默认研究模型保留 P2。V3-4 的 `risk_control` 在三个目标覆盖率上均通过预声明专长检验，
可以作为可选风险决策层。P2-LC 的自然缺失 Macro-F1 从 0.577684 提高到 0.578088，增益仅
+0.000404、胜出 2/5 折，未达到 +0.010 且至少 3/5 折胜出的冻结标准，因此不升级为发布模型。
V4-1 的 15/15 单元、106 项回归测试和完整性验收均通过；验收通过不改变首要科学假设失败的结论。

完整的项目路线、模型原理、逐阶段结果和面试解释见
[`docs/实验全流程与最终结论.md`](docs/实验全流程与最终结论.md)。

V1 原有的 60 个正式可信 run 与 6 个开发消融仍严格分开；不能把 66 个单元全部描述为正式
五折实验。所有后续版本同样保留协议锁、validation/test 隔离和正负结果。

官方 ExtraSensory 数据已完成审计，共 60 名用户、377,346 个样本。五个外层 fold 均采用仅训练集拟合的预处理，训练/验证/测试用户互斥，未知标签使用掩码，阈值仅由验证集选择，受控故障场景保持确定性。确定性合成数据仅用于工程冒烟测试，不参与研究结果。

项目使用原生 Windows 开发环境。Phase 0 已在 Python 3.12、PyTorch CUDA 12.8、NVIDIA RTX 2060、pytest、Ruff 和完整合成冒烟流水线上通过验证。

Phase 6 的 Streamlit 离线回放会校验并加载真实可信检查点、对应 fold 的预处理器、验证集阈值、模式定义及真实测试行。Phase 7 最终审计已在现有 `.venv-dev` 中通过；按项目所有者要求，本阶段不制作 PPT，也不新建或复现干净虚拟环境。

## 注册实验结果

最强结论是鲁棒性提升，而不是干净数据上的大幅增益：

- 受控完整模态条件下，Robust Gated Fusion 的 Macro-F1 最高：**0.6043**。
- 自然缺失条件下，Quality-Aware Fusion 的 Macro-F1 最高：**0.5781**。
- 自然缺失下，Quality-Aware 相对普通 Gated Fusion（**0.5670**）提升 **+0.0110**，并在全部 **5** 个外层 fold 中获胜。
- 随机移除三个模态时，Quality-Aware 的平均相对 Macro-F1 下降为 **9.45%**，普通 Gated Fusion 为 **12.87%**。
- 随受控污染增强，可靠度估计会一致下降，但学习到的融合权重不会始终降低受损模态的权重，因此相应可解释性假设仅得到部分支持。

这些是五折、单种子的描述性结果，不构成最先进性能声明。以上每个数值结论都可通过 `reports/readme_source_map.csv` 定位到表格筛选条件和源 run ID。完整 E1–E8 分析及限制见 `reports/technical_report.md`。

## 快速开始（Windows）

已准备好的环境为 `.venv-dev`。在 PowerShell 中执行：

```powershell
. .\scripts\activate_windows.ps1
python scripts\verify_environment.py
powershell -File scripts\run_smoke.ps1
```

在另一台装有 Python 3.12 的 Windows 计算机上重建环境：

```powershell
.\scripts\setup_windows.ps1
. .\scripts\activate_windows.ps1
```

安装脚本默认安装带 CUDA 12.8 的 PyTorch 2.11；不需要 CUDA 时使用 `-TorchVariant cpu`。精确验证版本见 `requirements/windows-dev.lock`。

单独执行冒烟命令：

```powershell
robustsense prepare --config configs/data/synthetic.yaml
robustsense train --model early --fold 0 --seed 13 --profile dev
robustsense evaluate --run-dir runs/synthetic-early-fold0-seed13 --suite smoke
pytest -q
ruff check .
```

## Phase 1 数据审计

模式探针会在扫描数据行之前输出完整真实表头及映射：

```powershell
.\scripts\fetch_extrasensory.ps1
.\scripts\run_phase1_audit.ps1 -ProbeOnly
.\scripts\run_phase1_audit.ps1
```

完整命令会验证每个源标签只能为 `0`、`1` 或 `NaN`，拒绝重复 `(user_id, timestamp)` 行，证明官方 fold 与生成划分相互隔离，并重建 `reports/data_audit/` 及模式、数据和划分清单。归档哈希是本地连续性检查，不是官方校验和。

## Phase 2 开发运行

```powershell
robustsense prepare --config configs/data/extrasensory.yaml --fold 0
.\scripts\run_phase2_dev.ps1
python scripts\run_phase2_sanity.py --fold 0 --seed 13
```

每个真实 run 都保存 Torch 检查点、训练日志、验证集派生的逐标签阈值、带掩码的验证/测试指标和 Parquet 预测。验收证据与限制见 `docs/PHASE2_REPORT.md`。

## Phase 3 开发运行

```powershell
.\scripts\run_phase3_dev.ps1
```

该脚本运行 B4、B5、仅分类损失的 P，以及带可靠度/一致性损失的 P。每个 run 还会在 `modality_diagnostics.parquet` 中保存逐样本模态权重、预测可靠度和人工故障元数据。详见 `docs/PHASE3_REPORT.md`。

## Phase 4 评估与报告

```powershell
.\scripts\run_phase4_dev.ps1
```

脚本评估注册模型矩阵，校验共享的受控样本 ID，并生成 E1–E8 表格、鲁棒性/质量响应图、完整性结果、来源映射和 `reports/technical_report.md`。重复生成报告时字节级稳定。协议、开发证据与限制见 `docs/PHASE4_REPORT.md`。

## Phase 5 正式注册实验

只检查、冻结并注册计划而不启动训练：

```powershell
python -m robustsense.cli.sweep --profile credible --project-root . --plan-only
```

继续尚未运行的可信单元，或仅重试失败单元：

```powershell
.\scripts\run_credible.ps1
.\scripts\run_credible.ps1 -RetryFailed
```

运行器不会重复执行已成功单元。每次尝试均记录在 `reports/run_attempts.csv`，详细日志位于 `runs/_phase5_logs/`。可信矩阵已经完成，因此可选的三种子 `full` 配置现已具备运行资格，但尚未执行。无需重新训练即可重建注册报告：

```powershell
robustsense report --project-root . --runs-dir runs --output-dir reports `
  --plan configs/evaluation/credible.yaml
```

另见 `docs/EXPERIMENT_PROTOCOL.md`、`docs/PHASE5_REPORT.md` 和 `reports/run_completeness.json`。

## Phase 6 离线回放演示

启动使用真实检查点的 Streamlit 沙箱：

```powershell
.\scripts\run_demo.ps1
```

页面支持选择官方 fold、随机或指定真实测试样本、B4/B5/P 检查点、模态移除、标准化特征空间中的 Gaussian/bias/scale 故障、干净与污染预测对比、质量/可靠度/权重诊断以及注册聚合结果。若产物缺失或不匹配，程序会停止，且没有伪造预测的回退路径。

这是对预计算数据特征的**离线回放**，不是实时传感器采集或物理传感器故障模拟器。详见 `docs/PHASE6_REPORT.md`。

## Phase 7 最终材料与审计

最终交付包括：

- `reports/technical_report.md`：包含相关工作、方法、E1–E8 结果、限制和结论的中文技术报告；
- `docs/ORAL_SCRIPT_2MIN.md` 与 `docs/ORAL_SCRIPT_8MIN.md`：两分钟和八分钟口头稿；
- `docs/DEFENSE_QA.md`：36 题答辩问题库与常见错误表述；
- `docs/ONE_PAGE_SUMMARY.md` 与 `reports/figures/robustsense_one_page.svg`：一页文字及视觉摘要；
- `docs/RELEASE_REVIEW.md`：数据、隐私、检查点、许可证和引用发布审查；
- `reports/audit_report.md`：现有环境最终审计证据与范围说明。

最终审计重新执行了 ExtraSensory 377,346 行全量审计、合成 prepare/train/evaluate、60/60 可信 run 报告生成、37 项测试、Ruff、`pip check` 和 Streamlit HTTP 健康检查。正式报告连续重建两次后比对 40 个文件，SHA-256 变化数为零。

按项目所有者的后续明确要求，PPT 和新虚拟环境复现不属于最终验收范围。公开发布前仍需选择许可证并在 `CITATION.cff` 中填写真实作者。

## 研究不变量

- 任务是多标签分类：使用 sigmoid 输出，绝不使用单一 softmax。
- 标签 `NaN` 表示未知，不计入损失或指标。
- 按用户划分；预处理、类别权重和阈值均不得拟合测试数据。
- 特征列必须依据真实表头/清单分组，禁止按位置切片猜测。
- 自然缺失评估和完整子集上的受控污染评估必须分开。
- 生成的图、表和演示必须可追溯到真实 run 产物。
- 不得把特征空间污染描述为物理传感器噪声。

完整项目规约见 `docs/PROJECT_SPEC.md`。

## 数据

官方 `ExtraSensory.per_uuid_features_labels.zip` 和 `cv5Folds.zip` 存放在 `data/raw/`。整个 `data/` 目录及生成的审计报告均被 Git 忽略。重新运行审计时，会先校验归档哈希再复用解压结果。

主要来源：

- [ExtraSensory 官方数据集页面](http://extrasensory.ucsd.edu/)
- [官方主数据 README](http://extrasensory.ucsd.edu/data/primary_data_files/README.txt)
- [官方入门教程](http://extrasensory.ucsd.edu/intro2extrasensory/intro2extrasensory.ipynb)
- [原始数据集论文（DOI）](https://doi.org/10.1109/MPRV.2017.3971131)
