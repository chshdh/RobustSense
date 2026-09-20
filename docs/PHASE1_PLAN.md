# Phase 1 实施与验收图

## 范围

Phase 1 只使用 ExtraSensory 官方预计算特征归档和官方五折用户列表，不处理绝对位置附加数据。

## 实施图

1. 记录归档来源、大小和本地计算的 SHA-256，并明确声明该哈希不是官方校验和。
2. 解压前检查每个 ZIP 成员，拒绝路径穿越、绝对路径、重复目标、符号链接和无效归档。
3. 从 `[UUID].features_labels.csv.gz` 文件名发现 UUID，并要求所有用户共享同一真实 CSV 表头。
4. 分类时间戳、全部 `label:*` 列、元数据、选用特征和明确忽略特征；每个输入特征必须且只能有一种归属。
5. 解析全部 20 个官方 fold 组件，验证补集关系和测试 fold 两两互斥，再生成确定性的外层测试/下一 fold 验证划分。
6. 分块扫描所有行；遇到非数值特征/标签、标签超出 `{0, 1, NaN}`、重复 `(user, timestamp)`、空文件或用户/fold 不匹配时失败。
7. 使用单一命令重建规约第 2.7 节要求的 CSV、JSON、PNG、Markdown、模式、数据和划分产物。

## 验收命令

```powershell
. .\scripts\activate_windows.ps1
.\scripts\fetch_extrasensory.ps1
.\scripts\run_phase1_audit.ps1 -ProbeOnly
.\scripts\run_phase1_audit.ps1
python -m pytest -q
python -m ruff check .
```

探针命令是审查门：它在昂贵的逐行审计前暴露完整真实表头与映射。只有官方归档上的全部结构合同通过，完整命令才可验收。
