# Phase 2 实施与验收图

## 冻结协议

- 外层测试 fold：0；验证 fold：1；训练/验证/测试用户数：36/12/12。
- Phase 1 的 177 列真实模式和 15 标签集在本阶段不可修改。
- 稳健中位数/IQR 统计量和类别权重只用训练用户拟合。
- 逐标签决策阈值只用验证预测调节。
- 测试数据只使用冻结训练统计量转换一次，拟合、早停、类别加权和阈值选择不得读取测试数据。
- Phase 2 只评估自然缺失；受控污染从 Phase 4 开始。

## 验收图

1. `prepare`：创建 fold 专属训练/验证/测试缓存、特征掩码、模态可用性、五个可部署质量特征和版本化预处理器。
2. `B0`：训练六个单模态模型；对应模态不可用的样本不计入损失或指标。
3. `B1`：训练线性早期融合基线。
4. `B2`：训练 MLP 早期融合基线。
5. `B3`：训练模态专属后期分类器，并按可用性掩码平均。
6. 每个 run 保存检查点、解析后配置、环境、源清单、训练日志、验证阈值/指标、测试指标和 Parquet 预测。
7. 单元检查证明未知标签梯度掩码、仅训练集缩放、常量列处理、仅验证集阈值和检查点合同检查正确。
8. 在不使用测试 fold 调参的前提下运行确定性标签打乱 sanity check。

## 命令

```powershell
. .\scripts\activate_windows.ps1
robustsense prepare --config configs/data/extrasensory.yaml --fold 0
.\scripts\run_phase2_dev.ps1
python -m pytest -q
python -m ruff check .
```

`dev` 配置仅提供工程证据，不得作为研究结论。
