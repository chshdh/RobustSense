# Phase 0 报告

日期：2026-09-10（Asia/Shanghai）

## 结果

Phase 0 项目骨架和开发环境已在原生 Windows 上实现并验证。项目所有者决定不使用 WSL。隔离环境 `.venv-dev` 包含数据、机器学习、报告、演示、测试和代码检查所需的完整依赖。

## 变更内容

- 打包：`pyproject.toml`、`LICENSE`、`CITATION.cff`、`.gitignore`。
- 配置合同：数据、模型、实验、污染和本地示例配置。
- 源码：常量、配置/I/O/种子工具、用户级划分、合成夹具、NumPy 线性冒烟基线、带掩码 BCE/F1、流水线和 CLI 模块。
- 测试：未知标签损失掩码、用户划分、确定性种子和端到端冒烟测试。
- 文档：README、带本地元数据的完整项目规约和构建状态。
- 脚本：WSL shell 与 PowerShell 冒烟运行器。

## 关键设计选择

- Phase 0 有意保持少依赖。仓库中的 `.yaml` 使用兼容 JSON 的语法，可由标准库解析；Phase 1 可在不改配置路径的情况下切换到 PyYAML。
- 合成夹具包含六个模态、15 个标签、自然模态缺失、未知标签和确定性的用户级训练/验证/测试划分。
- NumPy 线性基线仅用于验证合同与产物流，不是 B1/B2 研究证据，后续必须由计划中的 PyTorch 实现取代。
- 所有合成结果均标记 `synthetic_only: true`；run 元数据同时记录 `wsl_native_validation: false`。

## 执行命令

```text
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m pip install -e ".[dev]" --no-build-isolation
python -m pytest -q
ruff format .
ruff check . --fix
powershell -File scripts/run_smoke.ps1
robustsense train --model gated --fold 0 --seed 13 --profile dev
```

最后一条是负路径检查，按预期返回退出码 1。

## 验证结果

- 可编辑包安装：通过。
- `prepare`、`train`、`evaluate` CLI 帮助：通过。
- 单元/集成测试：`6 passed`。
- Ruff：`All checks passed`。
- 仓库冒烟脚本：prepare、单 epoch 训练、evaluate、pytest 和 Ruff 全部通过。
- 负路径合同：不支持的 Phase 0 模型返回非零退出码。
- 依赖一致性：`pip check` 通过。
- CUDA：PyTorch 2.11.0+cu128 检测到 RTX 2060，并成功完成 GPU 张量运算。

## 生成产物

`runs/synthetic-early-fold0-seed13/` 包含解析后配置、环境、划分/数据清单、训练日志、检查点、阈值、验证/测试指标、预测和 run 摘要。`data/manifests/` 与 `data/processed/` 包含合成夹具及其清单。上述顶层目录均被 Git 忽略。

## 剩余风险与 Phase 1 前置条件

1. 替换 `docs/PROJECT_SPEC.md` 中的占位所有者名称。
2. 手动获取两个 ExtraSensory 官方归档并放入 `data/raw/`。
3. 在检查真实归档 README/教程及 CSV 表头前，不得开始制定 Phase 1 模式规则。
