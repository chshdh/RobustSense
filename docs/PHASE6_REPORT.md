# Phase 6 报告：离线回放与传感器故障沙箱

## 结果

Phase 6 已在原生 Windows 项目环境中实现并通过验收。Streamlit 应用加载真实的 Phase 5 可信产物并回放真实 ExtraSensory 测试记录。它不会生成回退预测，也不会冒充实时采集系统。

在 PowerShell 中启动：`./scripts/run_demo.ps1`

默认地址为 `http://localhost:8501`。

## 产物合同

演示只提供可信配置中的 B4 Gated、B5 Robust Gated 和 P Quality-Aware 检查点。fold 与模型选择会解析为一个精确 run ID。

推理前，`robustsense.demo.load_demo_bundle` 会检查：

- `best_checkpoint.pt`；
- `resolved_config.yaml`；
- 具有 `source_split=val` 的 `thresholds.json`；
- `evaluation_manifest.json`；
- 对应 fold 的 `processed_manifest.json`；
- `preprocessor.json` 和 `preprocessor.npz`；
- 对应的 `test.npz`；
- 检查点模型、阶段、标签、模态维度、处理清单哈希、预处理器哈希和源模式哈希。

产物缺失或不匹配时会抛出 `DemoArtifactError`，页面停止并显示可执行的报告重建命令。系统没有合成数据、随机概率或硬编码预测回退。

## 界面

页面提供：

1. 项目范围、当前 run ID、检查点身份、fold、种子、设备和明确的离线回放警告；
2. 确定性索引选择或随机真实测试样本选择；
3. 禁用当前可用模态的控件；
4. 使用冻结评估强度的标准化 Gaussian、bias 和 scale 特征空间故障控件；
5. 干净/当前标签概率、验证阈值、决策和已知目标；
6. 逐模态可用性、五个可观测质量值、可选可靠度和融合权重；
7. 使用 B4、B5、P 评估同一样本与场景；
8. 注册的五折模态缺失聚合曲线，使单样本展示具有总体上下文；
9. 明确说明门控权重是竞争性分配权重，不是经过校准的可靠度概率。

匿名用户 ID 在显示前会再次哈希，应用不会暴露原始 UUID。

## 不变量

- 至少保留一个自然可用模态。
- 目标污染模态不能同时被禁用。
- 禁用模态的融合权重精确为零。
- 概率来自 `sigmoid(logits)`，决策使用所选 run 保存的验证阈值。
- 同一页面比较的模型使用相同 fold、样本索引、用户哈希、时间戳和污染控件。
- Gaussian 噪声使用确定性场景键，且只改变标准化预计算特征。
- 页面明确说明它不是实时采集或物理故障模拟器。

## 验证

真实产物冒烟测试在 CPU 上加载 fold 0 的 B4、B5 和 P 检查点，选择同一条六模态完整测试记录，禁用 audio，对 phone-state 特征施加强度 2.0 的 Gaussian 污染，并验证：

- 各模型的匿名用户、时间戳和样本索引一致；
- 全部标签概率有限；
- audio 可用性为 false；
- 每个模型的 audio 融合权重精确为 0.0。

Streamlit 服务已在临时本地端口成功启动，`/_stcore/health` 返回 HTTP 200，检查后临时服务已停止。两个演示专用单元测试通过，Ruff 接受演示模块、应用和测试。

## 限制

界面回放分钟级预计算数据，不执行原始特征提取、传感器同步、流处理、移动部署或物理故障仿真。单个样本只用于说明，注册聚合结果才是研究证据。
