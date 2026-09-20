# RobustSense 模型卡

## 模型族

RobustSense 是面向多标签上下文识别的质量感知多源融合研究实现。最终注册比较包括六个单模态 MLP、线性早期融合、非线性 MLP 早期融合、后期融合、带可用性掩码的 Gated Fusion、污染训练的 Robust Gated Fusion，以及 Quality-Aware Robust Fusion。

Quality-Aware 模型使用模态专属编码器、可观测质量特征、辅助 ReliabilityNet 输出、带掩码的学习融合权重、Sensor Dropout、特征空间噪声增强和干净/污染预测一致性正则化。

## 输入

每个样本包含分属六个模态的 177 个标准化预计算特征、六元素可用性掩码和逐模态可观测质量特征。输入由与检查点对应的 fold 专属预处理器生成。特征顺序必须来自保存的模式和模态切片，不能假设位置布局。

检查点只能与匹配的 `resolved_config.yaml`、fold 专属预处理数组及元数据、`thresholds.json`、处理数据/划分清单和标签/模态模式共同加载。

## 输出

分类器针对 15 个非互斥上下文标签输出 sigmoid 概率，决策使用验证集选定的逐标签阈值。训练和评估均会屏蔽未知目标标签。

门控模型输出逐模态融合权重，Quality-Aware 模型还输出预测可靠度。不可用模态的融合权重被硬掩码为精确的零，可用模态权重之和为一。

融合权重不是正确概率，也不能作为因果解释。注册实验表明，ReliabilityNet 输出会随受控污染单调响应，但门控权重在 Gaussian 噪声和 bias drift 下不会始终下降。

## 训练

可信协议采用用户独立外层 fold、种子 13、批量大小 512、AdamW、最多 80 个 epoch、以验证 Macro-F1 为依据且 patience 为 10 的早停、梯度裁剪、带掩码加权 BCE，以及验证集派生的逐标签阈值。污染由 run 种子、epoch、样本身份和视图共同确定。

Quality-Aware 训练可包含分类、可靠度和一致性损失。人工故障元数据只作为训练监督，推理时不需要。

## 注册评估

五折可信矩阵的 60 个单元全部通过 run 输出合同。Robust Gated Fusion 在受控完整模态上的平均 Macro-F1 最高（0.6043）。Quality-Aware Fusion 在自然缺失下最高（0.5781），比普通 Gated Fusion 高 0.0110，并在全部五个配对外层 fold 中获胜。

随机移除三个模态时，Gated、Robust Gated 和 Quality-Aware 的平均相对 Macro-F1 下降分别为 12.87%、10.33% 和 9.45%。这些是五折、单种子的描述性结果，不能证明统计显著性、跨数据集泛化或最先进性能。

权威证据位于 `reports/run_completeness.json`、`reports/tables/aggregate_performance.csv`、`reports/tables/e4_drop_random_k.csv`、`reports/tables/e7_response_correlations.csv`、`reports/technical_report.md` 和 `reports/source_map.csv`。

## 预期用途

适用场景包括：缺失模态和污染鲁棒性的离线研究、小型多模态融合架构的可复现比较、在真实 ExtraSensory 测试样本上检查模型行为，以及无泄漏多标签评估的教学示例。

## 超出范围及禁止解释

模型未针对实时手机采集、原始信号推理、跨数据集部署、当前设备人群、医疗或安全关键用途、身份推断或个人监控完成验证。离线演示不得被描述为已部署的移动系统。

Gaussian、bias 和 scale 场景只修改标准化预计算特征，不是物理传感器故障模拟。完整模态受控子集存在选择偏差，不能替代自然缺失评估。

## 性能与效率边界

Quality-Aware 检查点包含 134,875 个可训练参数，大小约 0.55 MiB。在本地评估环境中，注册的批量模型前向延迟约为每样本 0.0134 ms。该数值不包括特征提取、文件 I/O、流处理、功耗和移动端运行开销，因此不是端到端部署指标。

## 限制与风险控制

- 可信结果使用五个用户 fold，但只有一个训练种子。
- A0–A5 是 fold 0 上的短期开发消融，不是五折因果证据。
- 稀疏和含混标签的 F1 明显低于常见结构性上下文。
- 在新人群或设备分布下，阈值和校准可能漂移。
- 可靠度输出对污染的响应比学习到的门控权重更直接。
- 污染感知训练的 CPU 开销较高；训练效率不属于模型质量结果。

使用者必须检查检查点合同，产物缺失或不匹配时立即停止。系统没有随机或硬编码预测回退路径。

## 引用与版本

本卡描述 RobustSense 0.1.0 及 SHA-256 为 `91a4e5fd7547ecf347d7fb3d7ee101d7389ac71f77386d4301e1bc1ec8929a7b` 的冻结 Phase 5 可信协议。复用前请查阅 `CITATION.cff` 和 ExtraSensory 原始引用。
