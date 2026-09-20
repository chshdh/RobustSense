# RobustSense 本地执行元数据

> 记录日期：2026-09-10（Asia/Shanghai）  
> 当前构建位置：`C:/Users/Cs060/Documents/Codex/2026-09-10/zi-a/outputs/robustsense`  
> 正式开发平台：Windows 11 10.0.26200；项目所有者已决定不使用 WSL  
> Python：3.12.14（满足仓库 `>=3.11` 约束）  
> 本地数据路径：仓库相对路径 `data/raw/`；官方特征包与 cv5Folds 已下载并做本地 SHA-256 记录  
> GPU：NVIDIA GeForce RTX 2060 6GB；PyTorch 2.11.0 + CUDA 12.8 已验证  
> 计划实验档：`credible`（5 folds × 1 seed）；`full` 仅在 credible 完整后考虑  
> 项目开始日期：2026-09-10  
> 负责人：待项目所有者补充姓名  
> 冻结标签：本规约第 2.2 节所列 15 标签  
> 冻结场景：`clean_complete`、`drop_each_one`、`drop_random_k`、`gaussian_noise`、`bias_drift`、`scale_error`  
> 当前阶段：Phase 4 双线评估、E1–E8、完整性检查和自动报告已完成 dev 验收；下一阶段为 Phase 5 正式实验

---

# RobustSense 可执行项目总规约

> 项目中文名：**面向传感器缺失与噪声干扰的多源异构信息自适应融合感知系统**  
> 英文名：**RobustSense: Quality-Aware Multi-Source Sensor Fusion for Context Perception under Missing and Corrupted Modalities**  
> 文档用途：项目立项、Coding Agent 执行、实验验收、README/技术报告撰写、导师交流和答辩准备。  
> 默认路线：ExtraSensory 预计算特征版 + 多标签分类 + 用户级五折验证 + 质量感知门控融合 + 传感器故障注入 + Streamlit 离线回放 Demo。

---

## 0. 一页结论

### 0.1 项目不是在做什么

这不是“把多个传感器特征拼起来做一次分类”，也不是“套一个 LSTM/Transformer 后只汇报准确率”。

本项目研究的问题是：

> 手机、手表、位置、音频和设备状态等信息源并不总是同时存在，也不总是可靠。模型怎样根据每个时刻的模态可用性和质量，自适应决定信任哪些传感器，并在传感器缺失或受污染时保持稳定的上下文感知能力？

### 0.2 最终应当交付什么

1. 一个可复现的 Python/PyTorch 仓库；
2. 一套无用户泄漏的数据处理与五折评估流程；
3. Single Sensor、Early Fusion、Late Fusion、Gated Fusion、Robust Gated Fusion、Quality-Aware Robust Fusion 六类模型；
4. 完整模态、自然缺失、缺失一个/两个模态、不同噪声强度下的实验；
5. 消融实验、逐标签结果、门控权重与可靠度可解释性分析；
6. 一个读取真实 checkpoint 和真实测试样本的 Streamlit “离线回放与故障注入”Demo；
7. README、数据卡、模型卡、实验报告、演示稿和导师追问题库；
8. 所有表格和图片由结果文件自动生成，不手填数字。

### 0.3 推荐技术栈

- Python 3.11；
- PyTorch；
- NumPy、Pandas、scikit-learn；
- PyYAML 或 OmegaConf（二选一，优先简单的 PyYAML）；
- Matplotlib、Seaborn；
- Streamlit、Plotly；
- pytest、ruff；
- TensorBoard 可选；默认实验记录使用本地 JSON/CSV/Parquet，不强依赖在线平台。

预计算特征版使用小型 MLP，不要求独立显卡。不要在第一版引入 Transformer、分布式训练、在线数据库或微服务。

### 0.4 真实性边界

- 可以让 Agent 生成绝大多数代码，但数据必须真实下载、程序必须真实运行、结果必须真实保存、失败实验也必须保留；
- 不能把“预计算特征的离线回放”描述为“手机实时采集”；
- 不能把人工生成的示例数字放进最终结果；
- 在完成系统性文献检索前，不声称方法是学术首创。建议表述为“本项目提出/实现的质量感知鲁棒融合方案”；
- 若最终方法没有优于基线，也要如实分析原因。一个设计严谨、结论可信的负结果仍然比伪造的高分项目更有研究价值。

---

## 1. 项目章程

### 1.1 研究对象

ExtraSensory 是真实日常环境下采集的多源行为上下文数据。官方页面说明其包含 60 名用户、超过 30 万个分钟级样本；每分钟约采集 20 秒，信息源涵盖手机加速度计、陀螺仪、磁力计、手表加速度、位置、音频 MFCC、手机状态等。传感器天然存在不可用情况，因此适合研究缺失模态与融合鲁棒性。

主路线只使用官方约 215 MB 的“features and labels”压缩包。不要一开始下载数 GB 到十余 GB 的原始信号。

官方资料：

- [ExtraSensory 官方主页](https://extrasensory.ucsd.edu/)
- [原始论文 DOI](https://doi.org/10.1109/MPRV.2017.3971131)
- [统一多模态、多标签模型论文 DOI](https://doi.org/10.1145/3161192)

### 1.2 核心研究问题

- **RQ1：** 多源融合在用户独立测试中是否优于单模态模型？
- **RQ2：** 普通 Early/Late/Gated Fusion 在自然缺失和受控传感器失效下会怎样退化？
- **RQ3：** Sensor Dropout 与特征空间噪声增强是否能降低退化速度？
- **RQ4：** 显式质量特征和可靠度估计器是否能让门控权重随故障变化，并进一步提升鲁棒性？

### 1.3 可证伪假设

- **H1：** 在完整模态条件下，多源融合的 Macro-F1 高于大多数单模态模型；
- **H2：** 模态数量减少或噪声增强时，普通融合方法的性能下降；
- **H3：** 在相同训练/测试划分下，带 Sensor Dropout 的模型比不带该增强的 Gated Fusion 具有更小的相对性能下降；
- **H4：** 在可检测的人工污染下，质量感知模型会降低被污染模态的平均融合权重，且该趋势与污染强度一致。

这些是假设，不是预设结论。最终报告必须允许“部分支持”或“不支持”。

### 1.4 范围

第一版必须完成：

- 预计算特征；
- 六个核心模态：phone accelerometer、phone gyroscope、watch accelerometer、location、audio、phone state；
- 15 个默认上下文标签的多标签预测；
- 用户级划分；
- 六类基线/方法；
- 模态缺失与标准化特征空间噪声实验；
- 自动报告和离线回放 Demo。

第一版明确不做：

- 重新处理 6–11 GB 的原始时序信号；
- 真正的手机端实时采集和端侧部署；
- 将位置绝对经纬度输入模型；
- 生成式缺失模态重建；
- 为追求“高级感”无依据地加入大模型或 Transformer。

可选扩展只能在主线验收后进行：原始 IMU 子集、跨数据集验证、时序建模、用户自适应、端侧推理。

### 1.5 完成与成功的区别

**项目完成标准：** 数据流程无泄漏、模型与实验真实运行、所有主要结果可复现、Demo 使用真实模型、局限性说明完整。

**期望的正向研究结果：** Proposed Method 在缺失/噪声场景下比 Gated Fusion 和 Early Fusion 有更小的 Macro-F1 下降，并且可靠度/权重响应符合预期。

不要把“必须提升 X%”设为工程验收条件，否则 Agent 容易为满足指标进行无原则调参或结果筛选。

---

## 2. 数据任务定义

### 2.1 必须保持为多标签任务

ExtraSensory 的一个样本可以同时具有 `SITTING`、`COMPUTER_WORK`、`LOC_home`、`OR_indoors` 等多个标签。因此主任务是 multi-label classification，而不是互斥的 single-class classification。

令：

- 样本为 \(n\)；
- 模态为 \(i \in \{1,\ldots,M\}\)；
- 标签为 \(l \in \{1,\ldots,L\}\)；
- \(x_{n,i}\) 是第 \(i\) 个模态的特征；
- \(a_{n,i}\in\{0,1\}\) 表示该模态是否可用；
- \(y_{n,l}\in\{0,1\}\) 是标签值；
- \(s_{n,l}\in\{0,1\}\) 表示该标签对该样本是否已知。

关键约束：标签中的 NaN 表示“不知道/未提供”，不是负类。训练损失和评估指标必须通过 \(s\) 忽略未知标签。

### 2.2 默认 15 标签

建议先固定以下标签，兼顾运动、姿态、场所、设备使用和日常活动：

```yaml
labels:
  - SITTING
  - LYING_DOWN
  - OR_standing
  - FIX_walking
  - OR_exercise
  - BICYCLING
  - SLEEPING
  - EATING
  - TALKING
  - COMPUTER_WORK
  - LOC_home
  - LOC_main_workplace
  - OR_indoors
  - OR_outside
  - IN_A_CAR
```

Agent 在数据审计后必须输出每个标签的：已知样本数、正样本数、负样本数、阳性比例、阳性用户数、各 fold 分布。若某 fold 某标签没有正例，不得悄悄删除；应在报告中说明，并按预先写明的规则决定是否更换标签集。

可接受的更换规则应在看测试性能前确定，例如：标签至少出现在 20 个用户中、全数据至少 1,000 个正例，并且至少 4 个官方 fold 含正例。更换后要更新配置和数据卡，不能只为提高结果删困难标签。

### 2.3 模态分组

默认六组：

| 模态键 | 含义 | 官方论文中预计算特征规模（用于审计，不作为盲目硬编码依据） |
|---|---|---:|
| `phone_acc` | 手机加速度计 | 约 26 |
| `phone_gyro` | 手机陀螺仪 | 约 26 |
| `watch_acc` | 手表加速度计 | 约 46 |
| `location` | quick + server-side 相对位置特征 | 约 17 |
| `audio` | 13 维 MFCC 的均值和标准差 | 约 26 |
| `phone_state` | App、充电、锁屏、Wi-Fi、时间等离散状态 | 论文实验表示约 34 |

不同版本文件的列名和数量可能有差异。Agent 必须先读取官方 README/教程和真实 CSV header，生成 `data_manifest.json`，然后通过配置中的前缀/正则表达式分组。每一个输入特征列必须满足：恰好归入一个模态，或被明确列入 `ignored_features`。禁止依赖列的固定位置切片。

可在 schema 配置中使用类似规则，但具体正则要以真实 header 为准：

```yaml
modalities:
  phone_acc: ["^raw_acc:"]
  phone_gyro: ["^proc_gyro:"]
  watch_acc: ["^watch_acceleration:"]
  location: ["^location:", "^location_quick_features:"]
  audio: ["^audio_naive:", "^audio_properties:"]
  phone_state: ["^discrete:"]
```

### 2.4 下载与存储约定

推荐让用户从官方页面点击下载：

- `ExtraSensory.per_uuid_features_labels.zip`；
- 官方 `cv5Folds.zip`。

目录：

```text
data/
├─ raw/                 # 原始下载，永不修改
│  ├─ ExtraSensory.per_uuid_features_labels.zip
│  └─ cv5Folds.zip
├─ extracted/           # 每用户 csv.gz 与 fold 文件
├─ interim/             # 审计与临时索引
├─ processed/           # 按 fold 处理后的缓存
└─ manifests/           # 文件列表、哈希、schema、统计
```

要求：

- `data/`、checkpoint 和大型 run 文件写入 `.gitignore`；
- 下载脚本必须支持“文件已存在则校验并跳过”；
- 若官方未提供 checksum，本地首次下载后计算 SHA-256 并保存；这只能证明后续未变化，不能冒充官方校验值；
- 解压时防止 Zip Slip，拒绝目标目录之外的成员路径；
- 不下载绝对经纬度，避免不必要的隐私风险；
- README 必须引用官方数据页面和原始论文。

### 2.5 用户级划分

绝对禁止把全部行随机拆分为 train/test。同一个用户的行为模式和设备特征会泄漏到测试集。

推荐使用官方 5-fold 用户划分：

- 最小可信实验：5 个 fold，每次一个 fold 作测试；
- 对每次外层 fold，从剩余训练用户中按固定规则留出验证用户；
- scaler、imputer、异常阈值、类别权重和决策阈值都只能使用训练/验证部分拟合；
- 测试 fold 只在训练完成后评估；
- 保存每个 fold 的 UUID 列表和哈希。

开发阶段可以指定：fold 0 测试、fold 1 验证、其余 fold 训练。最终五折时，每个外层 fold 的验证用户使用确定性轮换或固定比例拆分，并把规则写入 `split_manifest.json`。

### 2.6 预处理协议

对每个外层 fold 独立执行：

1. 读取训练用户；
2. 记录原始 finite/NaN mask；
3. 只用训练集计算每列中位数、IQR（或均值/标准差）和异常阈值；
4. 默认使用 robust scaling：\((x-\mathrm{median})/\mathrm{IQR}\)，IQR 为 0 时使用 1；
5. 对标准化后的 NaN 填 0；
6. 保留 feature-level mask，计算 modality availability 和 quality features；
7. 验证集、测试集只应用训练统计；
8. 没有任何可用模态，或 15 个目标标签全部未知的样本才可剔除，并记录数量。

不要先对全数据标准化再划分。不要为了得到“完整数据”而把自然缺失行全部删除。

### 2.7 数据审计必须产出

```text
reports/data_audit/
├─ dataset_summary.json
├─ feature_schema.csv
├─ modality_missingness.csv
├─ label_prevalence.csv
├─ label_by_fold.csv
├─ user_sample_counts.csv
├─ modality_coavailability.png
├─ label_cooccurrence.png
└─ audit_report.md
```

`audit_report.md` 至少回答：样本/用户/列数是否符合预期、六组模态各多少列、自然缺失比例、标签是否为 0/1/NaN、时间戳是否唯一、每个 fold 是否含足够正例、哪些行被剔除以及原因。

---

## 3. 模型设计

### 3.1 统一接口

所有模型使用同一 batch 结构，便于公平评估：

```python
batch = {
    "features": {modality: FloatTensor[B, D_i]},
    "feature_masks": {modality: BoolTensor[B, D_i]},
    "availability": BoolTensor[B, M],
    "quality_features": FloatTensor[B, M, Q],
    "targets": FloatTensor[B, L],
    "target_mask": BoolTensor[B, L],
    "user_id": list[str],
    "timestamp": LongTensor[B],
}
```

所有模型输出：

```python
output = {
    "logits": FloatTensor[B, L],
    "fusion_weights": FloatTensor[B, M] | None,
    "reliability": FloatTensor[B, M] | None,
}
```

### 3.2 模态编码器

六个模态有不同输入维度，但都投影到公共维度 \(d=64\)：

```text
[imputed features, feature-present mask]
        ↓
Linear(input_dim × 2, 128)
        ↓ GELU → LayerNorm → Dropout(0.2)
Linear(128, 64)
        ↓
h_i
```

编码器不共享参数，因为不同模态的物理意义和统计分布不同。模型大小要通过配置控制，并在结果中报告参数量。

### 3.3 可观测质量特征

质量向量只能来自部署时可获得的信息，不能把人工注入的“故障标签”直接作为模型输入。建议每个模态计算：

- `availability`：是否至少达到最低有效特征比例；
- `observed_fraction`：有限特征占比；
- `outlier_fraction`：相对训练集 robust 统计，\(|z|>4\) 的特征占比；
- `mean_abs_robust_z`：截断后的平均绝对 robust z-score；
- `max_abs_robust_z`：截断后的最大绝对 robust z-score；
- 可选 `temporal_jump`：同一用户相邻且时间差不超过 90 秒时，与上一样本的标准化距离。若存在时间断点则置为未知并传 mask。

MVP 可以先用前五项。`temporal_jump` 只作为扩展，因为 ExtraSensory 时间序列存在间断。

### 3.4 Quality-Aware Robust Fusion

每个模态：

\[
h_i = E_i(x_i, m_i)
\]

可靠度估计器：

\[
r_i = \sigma(R_i([h_i,q_i]))
\]

门控分数：

\[
e_i = G([h_i,q_i,r_i])
\]

不可用模态在 softmax 前设为负无穷：

\[
w_i = \operatorname{masked\_softmax}(e_i, a_i)
\]

融合：

\[
z = \sum_i w_i h_i
\]

输出：

\[
\hat{y}=C(z)
\]

推荐网络：

```text
ReliabilityNet: Linear(64+Q, 32) → GELU → Linear(32, 1) → Sigmoid
GateNet:        Linear(64+Q+1, 32) → GELU → Linear(32, 1)
Classifier:     Linear(64, 128) → GELU → Dropout(0.2) → Linear(128, L)
```

严格测试 masked softmax：所有不可用模态权重必须为 0；可用模态权重和必须约等于 1；每个样本至少保留一个可用模态。

### 3.5 训练损失

分类主损失为 masked weighted BCE：

\[
\mathcal L_{cls}=
\frac{\sum_{n,l}s_{n,l}\,\mathrm{BCEWithLogits}(o_{n,l},y_{n,l};\,p_l)}
{\sum_{n,l}s_{n,l}}
\]

其中每个标签的 `pos_weight` 只从训练集已知标签计算，并设置合理上限（如 20），避免极少数类导致梯度爆炸。

研究版可加入两项辅助损失：

1. **可靠度监督**：训练时知道人工污染的类型与强度，但该监督值不作为推理输入。令 clean=1、missing=0、其他污染按预先固定映射得到 \(r_i^*\)，使用 MSE 或 BCE；
2. **一致性损失**：同一样本干净视图与受污染视图的预测不应无故剧烈改变。以干净分支 `detach` 后的 sigmoid 概率作为软目标，只对已知标签计算 binary KL 或 MSE。

总损失：

\[
\mathcal L=\mathcal L_{cls}+\lambda_{rel}\mathcal L_{rel}+\lambda_{con}\mathcal L_{con}
\]

建议从 `lambda_rel=0.1`、`lambda_con=0.1` 开始，但必须通过验证集或小规模消融确认。主线先跑通只有 \(\mathcal L_{cls}\) 的版本，再增加辅助项。

### 3.6 必须实现的比较模型

| ID | 模型 | 目的 |
|---|---|---|
| B0 | Single Sensor | 判断每个信息源独立能力；六个模型 |
| B1 | Logistic/Linear Early Fusion | 给出传统浅层基线 |
| B2 | MLP Early Fusion | 全部特征与 mask 直接拼接 |
| B3 | Late Fusion | 各模态独立输出 logits，对可用模态平均或验证集加权 |
| B4 | Gated Fusion | 模态编码器 + 内容门控，不做故障增强 |
| B5 | Robust Gated Fusion | B4 + Sensor Dropout/噪声增强 |
| P | Quality-Aware Robust Fusion | B5 + 显式质量特征 + ReliabilityNet；最终方法 |

B2–P 应共享尽可能一致的隐藏维度、训练轮数、优化器、早停规则与标签阈值选择。报告参数量和推理时间，避免把更多参数造成的提升误称为融合机制优势。

---

## 4. 传感器故障与污染协议

### 4.1 两条评估线必须同时保留

**A. 自然缺失评估**：使用所有合法测试样本，保留数据本来的缺失模式，反映真实数据条件。

**B. 受控故障评估**：先从测试 fold 中筛出六模态均可用的样本，再对同一批样本注入不同故障。这样不同故障点使用完全相同的样本集合，性能变化才可归因于故障。但要明确完整模态子集可能产生选择偏差。

最终报告不能只选更好看的一条。

### 4.2 训练时增强

对当前原本可用的模态进行增强，并确保至少保留一个：

```yaml
corruption_train:
  view_probability: 0.8
  missing_count_probs: {0: 0.40, 1: 0.40, 2: 0.20}
  noisy_modality_probability: 0.30
  noise_types: [gaussian, bias, scale]
  gaussian_sigma_range: [0.10, 1.00]  # 标准化特征空间
  bias_range: [-0.75, 0.75]
  scale_range: [0.70, 1.30]
  keep_at_least_one: true
```

增强 RNG 必须由 run seed 和 batch/sample 标识确定，测试场景必须完全确定性。不要对标签做任何污染。

### 4.3 受控测试场景

1. `clean_complete`：六模态完整；
2. `drop_each_one`：分别关闭每一个模态；
3. `drop_random_k`：随机关闭 \(k=1,2,3\) 个模态；每个 k 使用至少 5 个固定 mask 或固定种子；
4. `gaussian_noise`：每次污染一个模态，\(\sigma\in\{0.25,0.5,1.0,2.0\}\)；
5. `bias_drift`：标准化空间偏移 \(b\in\{0.25,0.5,1.0\}\)；
6. `scale_error`：倍率 \(c\in\{0.5,0.75,1.25,1.5\}\)；
7. 可选 `mixed_failure`：一个模态缺失、另一个模态受噪声污染。

由于主路线使用的是预计算特征，以上是**特征空间故障模拟**，不能声称等价于真实物理传感器噪声。报告中将其列为局限；原始信号故障建模属于扩展。

### 4.4 退化指标

除了故障后的绝对 Macro-F1，还要报告：

\[
\Delta F1 = F1_{clean} - F1_{corrupt}
\]

\[
\mathrm{RelativeDrop}=\frac{F1_{clean}-F1_{corrupt}}{F1_{clean}+\epsilon}
\]

以及噪声曲线面积 `robustness AUC`。这样完整数据表现略低、但退化更慢的模型也能被公平讨论。

---

## 5. 训练和评估协议

### 5.1 默认训练配置

```yaml
seed: 13
batch_size: 512
epochs: 80
optimizer: adamw
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip_norm: 5.0
early_stopping:
  metric: val_macro_f1
  patience: 10
  mode: max
thresholding:
  method: per_label_f1
  search_grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

这些是起点而不是神圣参数。任何调参必须只看训练/验证结果，并记录搜索空间。测试结果不能反过来决定超参数或阈值。

### 5.2 指标

主指标：

- Macro-F1（只对该标签已知的样本计算）；
- Micro-F1；
- mAP / Average Precision；
- 每标签 F1、precision、recall、support；
- 可选 AUROC（仅在正负样本都存在时计算）。

校准与解释性指标：

- Brier score 或 Expected Calibration Error；
- 人工污染模态可靠度预测的 MAE/AUROC；
- 污染强度与该模态平均权重的 Spearman 相关；
- 不可用模态权重是否恒为 0。

Accuracy 不作为主指标。多标签“subset accuracy”过于苛刻且不直观，可作为附录。

### 5.3 阈值

- 每标签阈值只在验证集选择；
- 将阈值随 checkpoint 保存；
- 所有测试场景使用同一组验证阈值，不针对故障场景重新调；
- 同时附上固定 0.5 阈值结果作为透明性检查。

### 5.4 实验规模

**开发档 `dev`：** 1 个 fold、1 个 seed、最多 5 epoch、可限制用户/样本，用于发现 bug，不用于报告结论。

**可信档 `credible`：** 官方 5 folds × 1 seed，所有模型，报告 fold 间均值和标准差。

**完整档 `full`：** 5 folds × 3 seeds。若算力有限，优先保证 5 folds，再给 B4/B5/P 增加多 seed。不要只挑最好 seed。

聚合时先在每个 fold 内聚合 seeds，再跨用户 fold 计算均值/标准差。不要把 15 个 fold-seed 结果错误地当成 15 个完全独立的数据集。

### 5.5 主实验矩阵

| 实验 | 数据集合 | 模型 | 输出 |
|---|---|---|---|
| E1 完整性能 | controlled complete subset | B0–P | 主指标、逐标签指标 |
| E2 自然缺失 | all legal test rows | B0–P | 主指标、按可用模态数分层 |
| E3 单模态失效 | same complete subset | B2–P | 每种 drop 的热力图 |
| E4 多模态失效 | same complete subset | B2–P | F1 vs missing count |
| E5 高斯噪声 | same complete subset | B2–P | 各模态噪声曲线 |
| E6 偏置/缩放 | same complete subset | B2–P | 鲁棒性曲线 |
| E7 质量响应 | same complete subset | B4–P | reliability/weight vs severity |
| E8 效率 | representative fold | B2–P | 参数量、延迟、模型大小 |

### 5.6 消融实验

以 P 为完整模型，依次移除：

- A1：无 Sensor Dropout；
- A2：无噪声增强；
- A3：无质量特征；
- A4：无 ReliabilityNet，只用内容门控；
- A5：无可靠度辅助损失；
- A6：无一致性损失；
- A7：共享编码器 vs 模态独立编码器；
- A8：uniform average 替代 learned gate。

至少必须完成 A1–A5。先在一个预定开发 fold 上筛选，核心结论再跑五折。不要在测试结果出来后临时增加只对自己有利的消融。

### 5.7 图表清单

必须自动生成：

```text
reports/figures/
├─ model_overview.png
├─ label_prevalence.png
├─ modality_missingness.png
├─ clean_performance.png
├─ natural_missingness_by_available_count.png
├─ f1_vs_missing_modalities.png
├─ drop_each_modality_heatmap.png
├─ noise_robustness_curves.png
├─ reliability_vs_noise.png
├─ gate_weight_vs_noise.png
├─ per_label_f1.png
└─ ablation.png
```

绘图脚本只读 `runs/**/metrics*.json|parquet`，不得在代码里写死论文数字。图中显示误差条、样本数和评估集合名称。

---

## 6. 仓库结构与文件职责

```text
robustsense/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml
├─ .gitignore
├─ configs/
│  ├─ data/extrasensory.yaml
│  ├─ model/{single,early,late,gated,robust_gated,quality_aware}.yaml
│  ├─ experiment/{dev,credible,full}.yaml
│  └─ corruption/{train,eval}.yaml
├─ data/                       # 不进 Git
├─ src/robustsense/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ download.py           # 可选下载/校验，失败时给人工下载指引
│  │  ├─ extract.py            # 安全解压
│  │  ├─ schema.py             # header 解析与模态分组
│  │  ├─ audit.py              # 数据审计
│  │  ├─ splits.py             # 官方用户 folds + val 规则
│  │  ├─ preprocessing.py      # fold-specific fit/transform
│  │  └─ dataset.py            # Dataset/Collate
│  ├─ corruption/
│  │  ├─ base.py
│  │  ├─ missing.py
│  │  ├─ noise.py
│  │  └─ scenarios.py
│  ├─ models/
│  │  ├─ encoders.py
│  │  ├─ fusion.py
│  │  ├─ reliability.py
│  │  ├─ baselines.py
│  │  └─ robustsense.py
│  ├─ training/
│  │  ├─ losses.py
│  │  ├─ metrics.py
│  │  ├─ thresholds.py
│  │  ├─ trainer.py
│  │  └─ checkpoint.py
│  ├─ evaluation/
│  │  ├─ evaluate.py
│  │  ├─ robustness.py
│  │  ├─ aggregate.py
│  │  └─ plots.py
│  ├─ cli/
│  │  ├─ prepare.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ sweep.py
│  │  └─ report.py
│  └─ utils/
│     ├─ config.py
│     ├─ logging.py
│     ├─ reproducibility.py
│     └─ paths.py
├─ app/
│  ├─ streamlit_app.py
│  ├─ inference.py
│  └─ components.py
├─ tests/
│  ├─ fixtures/                # 小型合成数据，不含真实隐私数据
│  ├─ unit/
│  └─ integration/
├─ scripts/
│  ├─ run_smoke.ps1
│  ├─ run_credible.ps1
│  └─ run_full.ps1
├─ docs/
│  ├─ PROJECT_SPEC.md
│  ├─ DATA_CARD.md
│  ├─ MODEL_CARD.md
│  ├─ EXPERIMENT_PROTOCOL.md
│  ├─ LIMITATIONS.md
│  └─ DEFENSE_QA.md
├─ runs/                       # 不进 Git，仅保留小型汇总可选
├─ reports/
│  ├─ figures/
│  ├─ tables/
│  └─ technical_report.md
└─ artifacts/                  # checkpoint，不进 Git 或用 release 管理
```

### 6.1 CLI 合同

最终至少支持：

```powershell
python -m robustsense.cli.prepare --config configs/data/extrasensory.yaml
python -m robustsense.cli.train --model gated --fold 0 --seed 13 --profile dev
python -m robustsense.cli.evaluate --run-dir runs/<run_id> --suite full
python -m robustsense.cli.sweep --profile credible
python -m robustsense.cli.report --runs-dir runs --output-dir reports
streamlit run app/streamlit_app.py
```

每条命令应有 `--help`、非零失败退出码和清晰错误信息。路径从配置/CLI 注入，不把某台电脑的绝对路径写入源码。

### 6.2 每个 run 的不可变产物

```text
runs/<run_id>/
├─ resolved_config.yaml
├─ environment.json
├─ split_manifest.json
├─ data_manifest.json
├─ train_log.csv
├─ best_checkpoint.pt
├─ thresholds.json
├─ val_metrics.json
├─ test_metrics.json
├─ robustness_metrics.parquet
├─ predictions.parquet
└─ run_summary.md
```

`environment.json` 包含 Git commit、Python/包版本、设备、开始/结束时间和 seed。`predictions.parquet` 至少包含匿名 UUID、timestamp、scenario、label、target、target_known、probability、prediction；必要时另存权重和可靠度。

---

## 7. 分阶段实施与验收

### Phase 0：项目骨架和契约

任务：

1. 建立上述目录、`pyproject.toml`、基础配置、日志和 CLI；
2. 写 `PROJECT_SPEC.md`，复制本规约的核心不可变约束；
3. 用合成数据建立最小端到端 smoke test；
4. 配置 ruff、pytest 和随机种子工具；
5. 写 `.gitignore`，排除数据、runs、checkpoint、缓存和环境变量。

验收：

- `pip install -e ".[dev]"` 成功；
- `pytest -q` 成功；
- 合成数据能在 CPU 上 1–2 分钟内完成 prepare → train → evaluate；
- 任一失败返回非零退出码；
- README 明确项目仍处于 scaffold 阶段，不展示虚构结果。

### Phase 1：真实数据解析与审计

任务：

1. 读取官方 README、教程和真实 header；
2. 解析每用户 `csv.gz`，UUID 从文件名获取；
3. 识别 timestamp、feature、label 列；
4. 建立六模态 schema；
5. 解析官方 5-fold 文件；
6. 生成第 2.7 节全部审计产物；
7. 为 schema 冲突、重复行、非法标签值、空 fold 编写失败检查。

验收：

- 用户数、样本量与官方描述数量级一致；
- 每个输入列恰好归入一组或明确忽略；
- 标签只允许 `{0,1,NaN}`；若发现其他值必须停止并报告；
- train/val/test 用户集合互斥；
- 数据审计报告由命令一键重建。

### Phase 2：无泄漏预处理和基线

任务：

1. 实现 fold-specific preprocessor；
2. 实现 masked BCE 和 masked metrics；
3. 实现 B0–B3；
4. 实现 per-label threshold tuning；
5. 在 dev profile 跑通；
6. 输出 predictions 和 test metrics。

验收：

- 一个有 NaN 标签的手工 batch 能证明 NaN 对 loss/metric 无贡献；
- 测试集极端值不会改变训练 scaler；
- 单模态不可用时正确跳过或输出明确定义；
- 同 seed 的短跑指标在容差内一致；
- random-label sanity test 接近随机表现，防止泄漏。

### Phase 3：门控、鲁棒增强和质量模型

任务：

1. 实现 B4；
2. 实现确定性 corruption registry；
3. 实现 B5；
4. 实现质量特征、ReliabilityNet 和 P；
5. 分步加入辅助损失；
6. 保存每样本权重、可靠度和人工故障元数据。

验收：

- 不可用模态权重严格为 0；
- 对每个样本，可用模态权重和为 1；
- 故障生成器不会关闭全部模态；
- 同一 scenario seed 产生相同 mask/noise；
- 推理时删除人工故障严重度字段，模型仍能运行；
- clean、missing、noise 三种合成场景都有单元测试。

### Phase 4：评估套件和自动报告

任务：

1. 实现自然缺失与 controlled complete 两条评估线；
2. 实现 E1–E8；
3. 生成聚合表格、误差条和鲁棒性曲线；
4. 实现 run 完整性检查：缺 fold、缺 seed、配置不一致则报告失败；
5. 自动生成 `technical_report.md` 的结果章节草稿。

验收：

- 所有模型在完全相同的测试样本 ID 上做受控比较；
- 任何图表都可追溯到 run ID；
- 聚合脚本不会把 NaN metric 当成 0；
- 报告同时显示绝对性能与相对下降；
- `report` 命令重复运行不改变结果。

### Phase 5：正式实验

顺序：

1. dev 单 fold 冒烟；
2. 一个预定 fold 的超参数/消融；
3. 冻结方法、标签、阈值规则和场景；
4. credible 五折全基线；
5. 若资源允许，full 三 seed；
6. 自动聚合；
7. 失败 run 只重跑失败单元，并记录原因。

验收：

- `run_registry.csv` 中每个计划 run 都有 `success/failed/skipped` 状态；
- 失败与重跑日志保留；
- 测试结果出来后没有未记录的超参数变化；
- 正文和图表使用最终冻结 run 集合。

### Phase 6：Streamlit Demo

定位：**离线测试样本回放与传感器故障注入沙盒**。

页面建议：

1. 顶部：项目问题、当前 checkpoint、fold、模型版本；
2. 左栏：选择真实测试样本、启用/关闭模态、噪声类型和强度、模型 B4/B5/P；
3. 中栏：Top 标签概率、阈值、预测变化；
4. 右栏：模态可用性、质量分数、可靠度、融合权重；
5. 下方：基线与鲁棒模型对比、当前样本之外的聚合实验曲线；
6. 明示“此页面不是实时采集，显示的是数据集样本离线推理”。

验收：

- checkpoint、preprocessor、threshold 和 schema 来自同一 run；
- 前端数字来自真实推理或真实结果文件；
- 缺文件时停止并给出可执行修复提示，不回退到假数据；
- 关闭模态后权重立即为 0；
- 可选择随机测试样本，不能只展示一个精选案例；
- 如提供“典型案例”，明确选择规则，并同时展示整体统计避免误导。

### Phase 7：文档、汇报和最终审计

任务：

1. README：问题、方法、快速开始、结果、局限、引用；
2. DATA_CARD：数据来源、用户划分、标签缺失、隐私；
3. MODEL_CARD：输入、输出、适用范围、不适用范围；
4. EXPERIMENT_PROTOCOL：预注册式写清全部实验规则；
5. 技术报告：摘要、引言、相关工作、方法、实验、结果、局限、结论；
6. 8–10 页演示稿；
7. 2 分钟和 8 分钟两个口头版本；
8. 逐项执行最终复现审核。

验收：新建干净虚拟环境，按照 README 从已下载数据开始，能完成 audit、smoke train、evaluation、report 和 Demo 启动。README 中展示的每个数字都能映射到结果文件。

---

## 8. 测试清单

### 8.1 单元测试

- schema：每列唯一归组、未知列触发错误；
- safe extraction：恶意 `../` 路径被拒绝；
- split：用户集合互斥；
- preprocessor：只 fit train，NaN 正确填充，常数列不除零；
- masked BCE：未知标签不产生梯度；
- metric：与手算小样本一致；
- corruption：至少一模态保留、相同 seed 可复现；
- masked softmax：不可用为 0，可用和为 1；
- threshold：只读取验证集；
- checkpoint：模型/config/schema/threshold 版本一致。

### 8.2 集成测试

- 20–100 行合成多用户数据完成全链路；
- 真实数据抽取 2–3 个用户完成审计和 1 epoch；
- 一个完整 dev fold 跑通 B2、B4、P；
- evaluation 生成所有预期文件；
- Streamlit inference 函数在无 UI 模式下完成一次预测。

### 8.3 科研有效性 sanity checks

- 打乱标签后性能应明显下降；
- 把用户 ID 或文件名意外作为特征时测试应失败；
- 训练集复制到测试集的故意泄漏 fixture 应被 split validator 捕获；
- 所有模态关闭必须报错而不是输出看似正常的预测；
- 重复同一 batch 的推理结果一致（eval mode）；
- 不使用测试集优化阈值的检查应有代码级断言或数据流隔离。

---

## 9. Coding Agent 工作规则

### 9.1 总规则

Agent 每次只执行一个 Phase，完成后输出：

1. 改动文件；
2. 关键设计选择；
3. 实际运行的命令；
4. 测试结果；
5. 生成的产物路径；
6. 未解决问题和下一 Phase 前置条件。

Agent 必须遵守：

- 先读 `docs/PROJECT_SPEC.md` 和当前配置，再写代码；
- 不修改已经冻结的实验协议，除非记录 ADR（Architecture Decision Record）和原因；
- 不硬编码本机路径；
- 不把 data/runs/checkpoint 提交到 Git；
- 不使用测试指标做超参数选择；
- 不生成或手填虚假实验结果；
- 报错时先定位最小复现，再修复，不用大范围重写掩盖问题；
- 保持每个 commit 可运行；
- 新行为必须有测试；
- 无法确认真实数据 schema 时暂停该局部实现并输出需要检查的 header，不猜列名。

### 9.2 可直接复制的总 Prompt

```text
你正在实现 RobustSense。请先完整阅读 docs/PROJECT_SPEC.md、configs/ 和现有测试，然后只执行我指定的 Phase。

不可变约束：
1. ExtraSensory 是多标签任务；标签 NaN 表示未知，必须在 loss 和 metric 中 mask，不能当 0。
2. 必须按用户划分，优先使用官方五折；preprocessor、class weight、threshold 不能使用测试信息。
3. 六模态为 phone_acc、phone_gyro、watch_acc、location、audio、phone_state。列分组基于真实 header/manifest，不按固定列位置猜测。
4. 保留自然缺失；受控故障实验只能在相同的完整模态测试子集上比较。
5. 所有实验数字、图和 Demo 必须来自真实 run/checkpoint；禁止 placeholder 冒充结果。
6. 主线使用预计算特征；特征空间噪声必须被准确描述，不能冒充物理原始信号故障。
7. 修改应小步、可测试、可复现。不要提前实现后续 Phase，也不要重构无关代码。

执行流程：
- 先检查仓库状态和相关文件；
- 写出本 Phase 的短计划和验收映射；
- 实现代码和测试；
- 运行最小测试，再运行本 Phase 验收命令；
- 若真实数据缺失，完成合成 fixture 可验证部分，并明确列出被阻塞项，不能伪造通过；
- 最后报告改动、命令、测试结果、产物和剩余风险。
```

### 9.3 分阶段 Prompt

**Phase 0：**

```text
执行 Phase 0：建立最小可安装仓库、配置系统、CLI、合成多用户多模态多标签 fixture 和端到端 smoke test。不要下载真实数据，不实现复杂模型。完成后必须证明 prepare→train(1 epoch)→evaluate 可在 CPU 跑通。
```

**Phase 1：**

```text
执行 Phase 1：基于已放入 data/raw 的官方 ExtraSensory 特征包和 cv5Folds，完成安全解压、真实 schema 探测、模态映射、用户折解析与数据审计。先展示 header 与映射结果，再固化规则。任何未归组或重复归组的特征必须显式报错。
```

**Phase 2：**

```text
执行 Phase 2：完成 fold-specific 无泄漏预处理、masked BCE/metrics、验证集逐标签阈值和 B0–B3。使用 dev profile 跑一个预定 fold，输出真实 run 目录。重点测试标签 NaN、用户隔离和 scaler 只 fit train。
```

**Phase 3：**

```text
执行 Phase 3：实现 Gated Fusion、确定性 Sensor Dropout/gaussian/bias/scale corruption、Robust Gated Fusion 和 Quality-Aware Robust Fusion。先只用分类损失跑通，再加入 reliability/consistency 辅助损失。为 masked softmax、故障可复现和至少保留一个模态写测试。
```

**Phase 4：**

```text
执行 Phase 4：实现完整评估套件。自然缺失使用全部合法测试行；受控故障使用同一完整六模态子集。生成 E1–E8 指标、predictions、聚合表和自动图，确保每个结果可追溯到 run_id。
```

**Phase 5：**

```text
执行 Phase 5：先检查实验协议是否冻结，再依次执行 dev、credible；full 仅在 credible 全部完整后执行。建立 run registry，失败任务单独重跑并保留日志。不要根据测试结果改阈值或挑 seed。
```

**Phase 6：**

```text
执行 Phase 6：构建 Streamlit 离线样本回放与故障注入 Demo。必须读取真实 preprocessor、checkpoint、threshold 和测试样本；任何依赖缺失就显示错误，不用随机/硬编码预测替代。页面必须明示并非实时采集。
```

**Phase 7：**

```text
执行 Phase 7：只基于最终 run registry 写 README、数据卡、模型卡、局限、技术报告和答辩材料。对 README 中每个数值建立 source mapping。最后在干净环境执行复现审核并输出 audit_report.md。
```

### 9.4 Agent 遇到问题时的决策顺序

1. 缩小到单元测试或最小数据样本；
2. 检查 schema、mask、shape、dtype 和 split；
3. 检查 train/eval mode、seed、threshold 和 checkpoint 对应关系；
4. 检查指标是否只用了 known labels；
5. 检查受控场景样本 ID 是否相同；
6. 记录失败原因后再做局部修复；
7. 只有架构契约确实错误时才写 ADR 并修改规约。

禁止通过删除测试、吞异常、跳过 fold、只展示成功 seed、把 NaN 变 0 或手改结果文件“解决”问题。

---

## 10. 建议开发节奏

### 10.1 五周稳健版

| 周 | 目标 | 可展示里程碑 |
|---|---|---|
| 第 1 周 | Phase 0–1 | 数据审计报告、用户折和模态缺失图 |
| 第 2 周 | Phase 2 | 单模态、Early、Late 基线真实结果 |
| 第 3 周 | Phase 3–4 | 门控、故障注入、鲁棒性曲线 |
| 第 4 周 | Phase 5 | 五折主结果和消融 |
| 第 5 周 | Phase 6–7 | Demo、技术报告、演示稿、答辩演练 |

### 10.2 两周压缩版

- 第 1–2 天：骨架、数据审计；
- 第 3–5 天：预处理和 B0–B4；
- 第 6–8 天：B5/P 与受控实验；
- 第 9–10 天：五折 credible；
- 第 11–12 天：Demo；
- 第 13–14 天：报告与复现审核。

压缩时可暂缓三 seed、偏置/缩放污染和 A6–A8，但不能省略用户级划分、masked labels、五折、真实结果和基本消融。

---

## 11. 风险清单与应对

| 风险 | 症状 | 应对 |
|---|---|---|
| 标签被当单分类 | softmax、每行只留一个标签 | 改为 L 个 sigmoid + masked BCE |
| 未知标签被当负类 | NaN fill 0 后直接算 loss | 单独保存 target mask |
| 用户泄漏 | 行随机拆分后分数异常高 | 官方用户 folds + split validator |
| 全数据预处理泄漏 | scaler 在 concat 全数据上 fit | 每 fold 保存独立 preprocessor |
| 模态列分错 | 维度不符或 phone state 混入其他组 | manifest + 唯一归组断言 |
| 完整子集偏差 | 完整模态结果异常乐观 | 同时报自然缺失与 controlled subset |
| 质量估计器学到注入标签 | 推理输入含 corruption severity | severity 仅作辅助监督/评估元数据 |
| 门控坍缩 | 一个模态始终权重接近 1 | 检查增强、熵、逐场景权重；不盲目强制均匀 |
| 类别不平衡 | 稀有标签 F1 为 0 | masked pos_weight、AP、逐标签报告 |
| Demo 造假感 | 显示固定 94% 或固定下降 | 读取真实样本/checkpoint，显示 run_id |
| 结果不可复现 | 同命令得到不同划分/场景 | 固定 seed、manifest、resolved config |
| 调参污染测试 | 看测试后改阈值 | 冻结协议，阈值只来自 val |
| 噪声叙事过度 | 把特征噪声说成真实硬件故障 | 明确“standardized feature-space corruption” |
| 项目过度膨胀 | 尚无基线就做原始信号/Transformer | 严格 Phase gate，扩展放最后 |

---

## 12. 最终报告建议结构

1. **摘要**：真实部署问题、方法、数据、实验设置、最主要真实结果；
2. **引言**：多源互补性与传感器不可靠性；
3. **相关工作**：行为上下文感知、多模态融合、缺失模态、可靠度估计；
4. **问题定义**：多标签、自然缺失、受控故障；
5. **方法**：模态编码、质量特征、可靠度、masked gate、增强和损失；
6. **数据与协议**：用户级五折、标签 mask、预处理、阈值；
7. **实验结果**：完整、自然缺失、人工故障、消融、解释性、效率；
8. **讨论**：何时有效、何时失败、门控是否真反映可靠性；
9. **局限**：预计算特征、特征空间模拟、数据年代/人群、domain gap、自报告标签噪声；
10. **结论与未来工作**：原始信号、在线质量估计、跨数据集和端侧部署。

不要用“达到了 SOTA”除非完成严格同协议文献比较。不要把不同标签集、不同 split 的论文数字直接横向比较。

---

## 13. 导师面谈表达

### 13.1 30 秒版本

> 我做的是多源异构传感器的鲁棒上下文感知。系统分别编码手机 IMU、手表、位置、音频和手机状态，再根据模态可用性与质量动态融合。和普通拼接不同，我重点评估了自然缺失、单/多传感器失效和不同噪声强度下的退化，并用 Sensor Dropout 和可靠度估计减少模型对单一信息源的依赖。数据按用户划分，任务保持为多标签，所有结果由统一五折实验生成。

### 13.2 2 分钟版本逻辑

1. 现实问题：传感器不是永远在线且质量稳定；
2. 数据：真实自然环境的手机/手表多源数据，天然有缺失；
3. 方法：模态独立编码 + 质量特征 + 可靠度估计 + masked gate；
4. 训练：随机关闭/污染模态，一致性与可靠度辅助监督；
5. 协议：用户级五折、自然缺失和相同样本受控故障双线；
6. 结果：只说真实跑出的数字，重点说性能下降斜率和权重响应；
7. 局限：当前是预计算特征上的离线研究，下一步是原始信号和真实设备故障。

### 13.3 必须会回答的问题

1. 为什么不能随机按行划分？——会产生 subject leakage。
2. 为什么是 sigmoid 而不是 softmax？——同一时刻多个上下文可同时成立。
3. 标签 NaN 是什么？——未知而不是负类，需 masked loss/metric。
4. 为什么需要模态独立编码器？——各模态维度、语义、尺度和缺失模式不同。
5. Early 与 Late Fusion 区别？——前者特征级联合，后者决策级组合；对缺失的处理和交互建模能力不同。
6. Sensor Dropout 为什么有效？——训练时打破对固定模态的依赖，让可用子集都得到优化。
7. 质量特征与 attention 有何区别？——attention 主要从内容学习相关性；本项目显式加入可用率和分布异常等可靠性线索。
8. 为什么不用准确率？——多标签且严重不平衡，Macro-F1/AP 更有解释性。
9. 怎么证明门控真在响应故障？——比较同一批样本随污染强度变化的权重、可靠度和性能。
10. 为什么完整模态子集和全样本都要报？——前者控制变量，后者更接近真实自然缺失，两者各有偏差。
11. 噪声模拟真实吗？——第一版仅是标准化特征空间压力测试，不等价于硬件物理噪声。
12. 最大局限是什么？——域差异、自报告标签噪声、旧设备数据、人工故障模型有限、尚未实时部署。
13. AI 写了多少？——如实说明 Agent 辅助工程实现；本人负责问题定义、协议、代码审查、实验与解释，并能从数据流到损失逐层讲清。

---

## 14. 最终交付清单

### 代码与复现

- [ ] 安装、测试和 CLI 均可用；
- [ ] 官方数据来源与引用完整；
- [ ] 数据不进 Git；
- [ ] 用户级 folds 可验证；
- [ ] masked labels 有测试；
- [ ] 六模态 schema 有 manifest；
- [ ] 六类模型接口一致；
- [ ] 故障场景确定性；
- [ ] run 产物完整；
- [ ] 干净环境复现通过。

### 实验

- [ ] Single/Early/Late/Gated/Robust/Proposed；
- [ ] 五折主结果；
- [ ] 完整模态与自然缺失；
- [ ] drop-one、drop-k；
- [ ] Gaussian noise；
- [ ] 至少 A1–A5 消融；
- [ ] 逐标签结果；
- [ ] 可靠度与门控响应；
- [ ] 参数量和推理时间；
- [ ] 所有图表自动生成。

### 展示

- [ ] Streamlit 显示真实 checkpoint/run_id；
- [ ] 随机测试样本可浏览；
- [ ] 可关闭模态和调噪声；
- [ ] 显示预测、质量、可靠度、权重；
- [ ] 显示聚合结果而非只展示个例；
- [ ] 明示离线回放与局限。

### 材料

- [ ] 中英文项目名和摘要；
- [ ] README；
- [ ] Data Card；
- [ ] Model Card；
- [ ] 6–8 页技术报告；
- [ ] 8–10 页演示稿；
- [ ] 2 分钟/8 分钟讲稿；
- [ ] DEFENSE_QA；
- [ ] 一页项目海报或架构图；
- [ ] 可公开的最小 checkpoint 或复现说明。

---

## 15. 推荐 commit 顺序

```text
chore: scaffold reproducible project and synthetic smoke test
feat(data): add safe extraction and ExtraSensory schema discovery
feat(data): add official user-fold parsing and audit report
feat(data): add fold-specific preprocessing and target masks
feat(model): add single-sensor and early/late fusion baselines
feat(train): add masked training metrics and validation thresholds
feat(model): add modality encoders and gated fusion
feat(robust): add deterministic sensor corruption suite
feat(model): add robust gated and quality-aware reliability fusion
feat(eval): add natural-missing and controlled robustness evaluation
feat(report): add result aggregation tables and figures
feat(app): add real-checkpoint Streamlit replay demo
docs: add reproducibility guide data card model card and limitations
test: add clean-environment end-to-end audit
```

每个 commit 都应通过当时已有的测试。不要把整个项目一次性生成成一个不可审查的大提交。

---

## 16. 第一条实际行动

把本文件复制为仓库的 `docs/PROJECT_SPEC.md`，在文件顶部补充：

- 机器与系统信息；
- 数据实际存放路径（只在本地配置，不提交）；
- 是否有 GPU；
- 计划使用 `credible` 还是 `full`；
- 项目开始日期；
- 负责人；
- 已冻结的默认标签和实验场景。

然后把第 9.2 节总 Prompt 与 Phase 0 Prompt 一起交给 Coding Agent。Phase 0 验收后再给 Phase 1，不要一次性让 Agent 同时实现所有阶段。

这套顺序的目的不是拖慢开发，而是让每一步都留下可验证证据。最终你面对导师时，不需要背诵一份“包装话术”；你可以沿着真实的数据、代码、实验和失败记录把整个研究逻辑讲清楚。
