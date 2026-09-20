# RobustSense V2 可执行项目总规约

> 项目中文名：**面向相关与持续传感器故障的可靠度约束多源融合感知系统**  
> 英文名：**RobustSense V2: Reliability-Constrained Multi-Source Sensor Fusion under Correlated and Persistent Failures**  
> 文档用途：V2 立项、Coding Agent 执行、实验冻结、阶段验收、面试展示和公开仓库整理。  
> 基线版本：RobustSense V1（Phase 0–7 已完成，60 个可信 run、6 个开发消融单元）。  
> 默认路线：保留 ExtraSensory 预计算特征和用户级五折协议，新增任务效用/可靠度解耦、可靠度约束门控、选择性预测、全模态组合与混合/持续故障评估。  
> 执行环境：继续使用现有 Windows `.venv-dev`；本版本不要求新建或复现虚拟环境，但必须保存环境核验结果。  
> 展示范围：不制作 PPT；交付 README、技术报告、Demo、两分钟/八分钟讲稿和面试问题库。

---

## 0. 一页结论

### 0.1 为什么要做 V2

V1 已经证明：污染感知训练能够减少模态缺失造成的相对退化，Quality-Aware 模型在自然缺失条件下取得最佳 Macro-F1；同时也发现 ReliabilityNet 会随污染增强而降低可靠度，但 Gate Weight 在 Gaussian 和 bias 条件下不会一致下降。

V2 不以“换一个更大的模型”为目标，而是围绕这个真实负面发现继续研究：

> 如何把传感器可靠度与任务相关的模态效用分开建模，让不可靠模态受到明确约束，并让系统在整体输入不可信时能够拒绝或降级，而不是始终强制输出预测？

### 0.2 V2 最终应当交付什么

1. 一个不破坏 V1 结果的独立 V2 配置、run 和报告空间；
2. 一个新的 P2 Reliability-Constrained Fusion 模型；
3. 任务效用、预测可靠度、最终融合权重三种语义明确分离的诊断输出；
4. 可靠度排序损失、校准指标和选择性预测；
5. 六模态全部 63 种非空可用组合评估；
6. 缺失+污染、相关故障和持续故障评估；
7. B4/B5/P/P2 的五折三种子核心矩阵；
8. 用户级配对 Bootstrap 置信区间和正式五折消融；
9. 一个能展示 `quality → reliability → utility → weight → prediction/abstention` 的真实检查点 Demo；
10. 面向面试的 README、架构图、性能优化记录、技术报告、讲稿和问题库。

### 0.3 不以正向结果作为工程验收条件

项目完成不要求 P2 必须超过 P，也不要求所有可靠度指标均改善。完成标准是：协议无泄漏、实现正确、比较公平、结果真实、统计完整、负面结果得到解释。若可靠度约束损害分类性能，应保留结果并分析任务效用与可靠度之间的冲突。

### 0.4 V2 明确不做

- 不重写 V1 数据管线和已经验收的基线；
- 不覆盖 V1 的 run、报告、协议锁和来源映射；
- 不因为“看起来高级”直接把全部模型换成 Transformer；
- 不把预计算特征污染描述为真实硬件故障；
- 不使用测试集选择可靠度阈值、拒绝阈值或超参数；
- 不把 Gate Weight 重新包装成纯可靠度；
- 不在核心 V2 验收前启动原始信号、第二数据集或手机端部署；
- 不制作 PPT，除非项目所有者后续重新明确要求。

---

## 1. 项目现状与版本边界

### 1.1 V1 权威事实

- 数据：60 名用户、377,346 条分钟级记录、177 个特征、六个模态、15 个多标签目标；
- 正式矩阵：12 个模型 × 5 个外层 fold × 1 个种子 13，共 60 个可信 run；
- 开发消融：A0–A5，共 6 个 fold 0/seed 13/三 epoch 单元；
- 受控完整 Macro-F1：Gated 0.5995、Robust Gated 0.6043、Quality-Aware 0.6022；
- 自然缺失 Macro-F1：Gated 0.5670、Robust Gated 0.5757、Quality-Aware 0.5781；
- 随机移除三个模态时的相对下降：12.87%、10.33%、9.45%；
- H4 仅部分支持：ReliabilityNet 对污染强度响应单调，门控权重响应不一致；
- V1 冻结协议 SHA-256：`91a4e5fd7547ecf347d7fb3d7ee101d7389ac71f77386d4301e1bc1ec8929a7b`。

以上数字只能由 V1 权威结果文件重建，不允许复制后手改。

### 1.2 V1 保护规则

V2 开发前必须：

1. 保存当前 `reports/run_completeness.json`、`reports/phase5_protocol_lock.json` 和 V1 结果摘要哈希；
2. 为 V1 状态创建版本说明；若 Git 历史允许，建立 `v1.0.0` 标签；
3. V2 run 使用独立 profile 和目录，不复用同名 run；
4. V2 报告输出到 `reports/v2/`；
5. V2 配置放入独立命名空间，建议 `configs/v2/`；
6. 若修改共享代码导致 V1 预测变化，必须先修复回归或记录 ADR，不能默认接受；
7. 只有哈希、配置、数据和检查点合同一致时，V2 才能复用 V1 的 B4/B5/P seed 13 结果。

### 1.3 V2 建议目录

```text
configs/v2/
├─ models/
│  ├─ reliability_constrained.yaml
│  └─ ablations/
├─ corruption/
│  ├─ train.yaml
│  ├─ exhaustive_masks.yaml
│  ├─ mixed_failures.yaml
│  └─ persistent_failures.yaml
└─ evaluation/
   ├─ dev.yaml
   ├─ credible_seed13.yaml
   └─ full.yaml

runs/v2/
reports/v2/
docs/v2/
```

`data/`、`runs/`、检查点和用户级派生产物继续保持 Git 忽略。

---

## 2. 研究问题、假设和贡献边界

### 2.1 核心研究问题

- **RQ1：** 将任务效用与模态可靠度解耦后，能否提高缺失和污染条件下的鲁棒性，同时保持完整输入性能？
- **RQ2：** 可靠度约束门控是否比仅把可靠度作为 Gate 输入更稳定地抑制受损模态？
- **RQ3：** 可靠度输出能否检测故障、反映污染强度并预测分类风险？
- **RQ4：** 系统在低可靠度条件下拒绝预测，能否改善被接受样本的风险—覆盖率权衡？
- **RQ5：** 在全部 63 种模态组合以及混合、相关、持续故障下，V1 结论是否仍成立？
- **RQ6：** 自然缺失模式本身是否携带标签信息，使模型产生 availability shortcut？

### 2.2 可证伪假设

- **H1：** P2 在自然缺失条件下的平均 Macro-F1 不低于 P，或具有更小的配对退化；
- **H2：** P2 在随机缺失两个/三个模态及 63-mask 最坏场景下的相对下降小于 P；
- **H3：** 对同一样本和目标模态，预测可靠度满足 `clean > mild > severe` 的比例高于 P；
- **H4：** P2 的受损模态最终权重与污染强度呈更稳定的负相关；
- **H5：** 以验证集确定的拒绝阈值筛选样本后，覆盖率下降会伴随被接受样本风险下降；
- **H6：** Mask-only 模型能够获得非随机表现，说明自然缺失模式含有一定上下文信息；若不成立，也必须如实报告。

所有假设允许“不支持”或“部分支持”。在看到可信测试结果后不得修改假设措辞。

### 2.3 预期贡献

1. **方法贡献：** 可靠度与任务效用解耦的约束门控；
2. **系统贡献：** 低可靠度条件下的选择性预测/拒绝机制；
3. **评估贡献：** 63 种可用性组合、混合与持续故障基准；
4. **实证贡献：** 五折三种子、用户级配对置信区间及正式消融；
5. **工程贡献：** 可恢复实验治理、真实检查点 Demo 和性能优化证据。

在完成同协议文献比较前，不声称 SOTA 或学术首创。

---

## 3. 数据与不可变实验约束

### 3.1 沿用 V1 数据任务

V2 默认不修改以下内容：

- 数据集和本地归档；
- 六模态定义与 177 特征模式；
- 15 个标签及其顺序；
- 标签 NaN 表示未知的语义；
- 官方用户五折；
- fold `f` 为测试、fold `(f+1) mod 5` 为验证、其余三个 fold 为训练；
- 每 fold 单独拟合训练中位数/IQR、类别权重和质量参考；
- 验证集选择逐标签阈值；
- 测试集只用于最终评估；
- 自然缺失与受控完整子集两条评估线。

### 3.2 输入与输出合同

P2 继续接收 V1 batch：

```python
batch = {
    "features": {modality: FloatTensor[B, D_i]},
    "feature_masks": {modality: BoolTensor[B, D_i]},
    "availability": BoolTensor[B, 6],
    "quality_features": FloatTensor[B, 6, 5],
    "targets": FloatTensor[B, 15],
    "target_mask": BoolTensor[B, 15],
    "user_id": list[str],
    "timestamp": LongTensor[B],
}
```

P2 输出扩展为：

```python
output = {
    "logits": FloatTensor[B, 15],
    "fusion_weights": FloatTensor[B, 6],
    "reliability": FloatTensor[B, 6],
    "utility_scores": FloatTensor[B, 6],
    "system_reliability": FloatTensor[B],
    "abstain": BoolTensor[B] | None,
}
```

训练时使用的故障类型、强度和 reliability target 只可作为监督/评估元数据，不可进入模型推理输入。

### 3.3 质量特征保持不变

每个模态继续使用五维质量向量：

1. availability；
2. observed fraction；
3. outlier fraction；
4. mean absolute robust z；
5. max absolute robust z。

V2 首版不增加 temporal jump，避免在主方法和时间建模之间混入额外变量。若后续加入，必须建立独立扩展协议。

---

## 4. P2 Reliability-Constrained Fusion

### 4.1 模态编码

沿用 V1 模态专属编码器：

\[
h_i=E_i(x_i,m_i),\qquad h_i\in\mathbb{R}^{64}
\]

编码器结构和隐藏维度默认与 P 相同，以减少参数量差异带来的混淆。

### 4.2 可靠度分支

\[
r_i=\sigma(R_i([h_i,q_i]))
\]

建议结构：

```text
Linear(64+5, 32) → GELU → Linear(32, 1) → Sigmoid
```

`r_i` 的语义限定为：当前模态相对训练参考是否可用且未明显受损。它不是标签预测概率，也不是最终融合权重。

### 4.3 任务效用分支

\[
u_i=U_i(h_i)
\]

建议结构：

```text
Linear(64, 32) → GELU → Linear(32, 1)
```

`u_i` 表示当前模态对分类任务的相对贡献，不要求随污染强度单调变化。

### 4.4 可靠度约束门控

定义正约束系数：

\[
\beta=\operatorname{softplus}(b)+\epsilon
\]

融合分数：

\[
s_i=u_i+\beta\log(r_i+\epsilon)
\]

权重：

\[
w_i=
\frac{a_i\exp(s_i)}{\sum_k a_k\exp(s_k)}
\]

等价地：

\[
w_i\propto a_i\exp(u_i)r_i^\beta
\]

必须满足：

- 不可用模态权重精确为 0；
- 可用模态权重和为 1；
- 所有模态不可用时立即失败，不产生看似正常的预测；
- `beta`、`epsilon` 和初始化写入 resolved config；
- 训练和推理使用同一公式。

### 4.5 融合与分类

\[
z=\sum_i w_i h_i
\]

\[
o=C(z),\qquad p=\sigma(o)
\]

分类器结构沿用 P，所有标签阈值只从验证集选择。

### 4.6 系统可靠度与选择性预测

系统级可靠度定义为：

\[
R(x)=\sum_i w_i r_i
\]

给定验证集确定的阈值 \(\tau\)：

\[
abstain(x)=\mathbb{1}[R(x)<\tau]
\]

主报告固定展示 90% 目标覆盖率对应的验证阈值，并同时输出完整 Risk-Coverage 曲线。测试集不得重新选择 \(\tau\)。拒绝表示“当前输入不足以支持可信输出”，不是预测负类。

### 4.7 训练损失

沿用：

- masked weighted BCE 分类损失 `L_cls`；
- 连续可靠度目标回归 `L_rel`；
- 干净/污染预测一致性 `L_cons`。

新增排序损失。对同一样本、同一模态的较干净视图 `a` 和较严重视图 `b`：

\[
L_{rank}=\max(0,m-r_a+r_b)
\]

总损失：

\[
L=L_{cls}+\lambda_{rel}L_{rel}+\lambda_{cons}L_{cons}+\lambda_{rank}L_{rank}
\]

默认起点：

```yaml
reliability_loss_weight: 0.1
consistency_loss_weight: 0.1
ranking_loss_weight: 0.05
ranking_margin: 0.1
beta_init: 1.0
epsilon: 1.0e-6
```

这些值只用于开发起点。最终可信配置必须在测试结果不可见的情况下冻结。

---

## 5. V2 故障与评估协议

### 5.1 三条评估线

1. **自然缺失：** 全部合法测试记录，保留真实 availability；
2. **受控单点故障：** 沿用 V1 clean/drop/noise 场景和相同完整测试总体；
3. **V2 扩展压力测试：** 63-mask、混合故障、相关故障和持续故障。

三条线必须分开报告，不得混合聚合为一个单分数。

### 5.2 全部 63 种非空模态组合

六模态共有：

\[
2^6-1=63
\]

种非空 availability mask。对每个 fold 的完整模态测试总体，固定样本 ID，并对 B4/B5/P/P2 使用完全相同的 63 个 mask。

至少报告：

- 63-mask 宏平均 Macro-F1；
- 按自然测试集 mask 频率加权的 Macro-F1；
- 最坏 mask Macro-F1；
- 每个可用模态数量下的平均与最坏结果；
- P2 相对 P/B5 的配对差异；
- 模态组合热图。

### 5.3 混合故障

第一版冻结为：

- 有序选择一个模态完全缺失；
- 从剩余五个模态选择一个施加 Gaussian；
- `sigma ∈ {1.0, 2.0}`；
- 共 `6 × 5 × 2 = 60` 个场景；
- 每个模型/fold 使用相同样本、目标模态和 RNG。

bias 与 scale 的混合场景可以在主矩阵完成后作为扩展，不能延迟核心验收。

### 5.4 持续故障

按用户和时间戳排序，只在相邻记录时间差不超过 90 秒时构建连续 episode。默认 episode 长度：

```yaml
episode_lengths: [5, 15, 30]
faults:
  - drop
  - gaussian_sigma_2
  - bias_1
```

报告：

- 故障前、故障中、恢复后的分类性能；
- 可靠度检测延迟；
- 恢复后可靠度返回基线范围所需步数；
- episode 不足时的排除数量和原因。

当前模型是逐样本模型，因此持续故障评估主要验证响应轨迹，不得声称模型已经进行时序推理。

### 5.5 Mask-only shortcut 诊断

实现只接收六维 availability 的轻量模型：

\[
\hat y=f(a_1,\ldots,a_6)
\]

并执行：

1. 用户级五折 Mask-only 基线；
2. 在测试用户内部确定性置换 availability；
3. 比较置换前后的 Macro-F1；
4. 分析 P/P2 是否依赖缺失模式捷径。

Mask-only 结果属于诊断，不与完整传感器模型直接宣称公平性能竞争。

---

## 6. 模型矩阵、run 数量和冻结顺序

### 6.1 V2 核心模型

| ID | 模型 | 用途 |
|---|---|---|
| B4 | Gated | 无故障增强的内容门控基线 |
| B5 | Robust Gated | 相同架构 + 故障增强 |
| P | Quality-Aware | V1 显式质量与 ReliabilityNet |
| P2 | Reliability-Constrained | V2 可靠度/效用解耦与约束门控 |

V1 的其余八个模型继续作为背景结果，不要求为所有 V2 扩展场景重新训练。

### 6.2 核心五折三种子矩阵

```yaml
folds: [0, 1, 2, 3, 4]
seeds: [13, 29, 47]
models: [gated, robust-gated, quality-aware, reliability-constrained]
```

总规模：

\[
4\times5\times3=60\text{ 个核心 run}
\]

若 V1 B4/B5/P seed 13 通过合同复用，则新增：

- B4/B5/P 的 seed 29、47：30 个；
- P2 的三个种子、五折：15 个；
- 共 45 个新增核心 run。

### 6.3 正式消融

| ID | 改动 |
|---|---|
| P2-A0 | 完整 P2 |
| P2-A1 | 移除可靠度约束，权重仅由 utility 决定 |
| P2-A2 | 移除 ranking loss |
| P2-A3 | 移除显式质量特征，ReliabilityNet 仅看内容 |
| P2-A4 | 移除选择性预测，仅报告强制预测 |

第一轮：A1–A4 × 五 fold × seed 13，共 20 个新增消融 run。根据验证阶段预先定义的选择规则，最多选择两个差异最大的变体补 seed 29、47，共最多 20 个新增 run。

不得根据测试集上“哪个结果更好看”选择补跑对象。选择规则应优先使用验证指标、可靠度诊断和计算预算。

### 6.4 运行顺序

1. 合成 fixture；
2. fold 0/seed 13/P2 dev；
3. fold 0/seed 13/P2-A1–A4 dev；
4. 冻结 P2 架构、损失、故障场景和评估指标；
5. P2 五折 seed 13；
6. P2-A1–A4 五折 seed 13；
7. B4/B5/P/P2 的 seed 29、47；
8. 统计聚合；
9. 扩展消融种子；
10. 最终报告和 Demo。

测试结果出现后，任何配置修改必须创建新协议版本和新 run ID。

---

## 7. 指标与统计协议

### 7.1 分类指标

- Macro-F1（主要）；
- Micro-F1；
- mAP；
- 每标签 precision/recall/F1/support；
- Brier Score；
- 完整条件与自然缺失条件的绝对性能。

### 7.2 鲁棒性指标

- `delta_macro_f1`；
- `relative_drop`；
- 缺失数量—性能曲线；
- robustness AUC；
- 63-mask 平均、频率加权和最坏 Macro-F1；
- 混合故障性能；
- 持续故障检测与恢复指标。

### 7.3 可靠度指标

- 污染强度与可靠度 Spearman 相关；
- 污染强度与最终权重 Spearman 相关；
- reliability target MAE；
- clean vs corrupt 故障检测 AUROC/AUPRC；
- 可靠度分箱校准曲线；
- 严重度排序正确率；
- 故障模态定位 Top-1/Top-k 准确率。

故障检测中使用 `1-r_i` 作为故障分数。缺失、污染、自然不可用需要分开报告，不能混成一个标签。

### 7.4 选择性预测指标

定义覆盖率：

\[
Coverage=\frac{\#accepted}{\#all}
\]

风险至少包括接受样本上的平均 masked BCE，并补充接受子集上的 Macro-F1/Micro-F1。输出：

- Risk-Coverage 曲线；
- 90% 覆盖率结果；
- 干净样本误拒绝率；
- 故障样本拒绝率；
- 自然缺失与受控故障分别计算。

### 7.5 聚合顺序

三种子正式结果：

1. 对每个 `fold × model × scenario` 聚合种子；
2. 再跨 fold 报均值和样本标准差；
3. 不把同一 fold 的不同种子当作不同用户数据集；
4. 所有模型比较使用相同 fold、seed、样本和场景。

### 7.6 用户级配对 Bootstrap

- 抽样单位为用户，不是数据行；
- 在每个 fold 的测试用户内有放回抽样；
- 推荐 2,000 次确定性重复；
- 对同一个 Bootstrap 样本同时计算两模型，得到配对差；
- 报告差值的 2.5%/97.5% 分位区间；
- 保存 bootstrap seed、用户抽样索引哈希和源 run ID。

五个 fold 均值不应被简单当作五个独立数据集执行未经说明的显著性检验。

### 7.7 预先声明的主要终点

主要终点按优先级冻结为：

1. P2 与 P 的自然缺失 Macro-F1 配对差；
2. P2 与 P 的随机移除三个模态相对下降差；
3. P2 与 P 的 63-mask 最坏 Macro-F1 差；
4. P2 与 P 的可靠度严重度排序正确率差；
5. P2 在 90% 覆盖率下的 selective risk。

其他指标为次要或探索性分析，避免对大量指标选择性汇报。

---

## 8. 性能工程计划

### 8.1 目标

V2 在新增正式 run 前必须先建立训练性能基线。目标不是强制 GPU 利用率达到某个百分比，而是定位端到端瓶颈并提高每秒样本数。

### 8.2 必测时间

- DataLoader 等待；
- batch 搬运；
- corruption；
- forward；
- backward/optimizer；
- validation；
- controlled evaluation；
- Parquet/JSON 写入。

### 8.3 允许的优化

- 批量张量化 corruption；
- 避免逐样本 Python/Pandas 循环；
- 缓存 fold 预处理数据和场景索引；
- `pin_memory`、`persistent_workers`、合理 `num_workers`；
- 在显存允许时增大 batch；
- 自动混合精度；
- 避免不必要的 CPU/GPU 同步；
- 报告吞吐、epoch 时间和峰值显存。

任何性能优化必须通过数值等价或容差测试。不得为了提速改变用户划分、故障定义、指标或样本集合。

---

## 9. V2 分阶段实施与验收

### Phase V2-0：冻结与保护

任务：

1. 保存 V1 权威产物哈希；
2. 建立 V2 目录和配置命名空间；
3. 写 V2 ADR 和实验协议锁；
4. 核验现有环境、GPU和依赖；
5. 建立性能基线；
6. 确认 V1 37 项测试及报告重建不回归。

验收：V1 结果未变；V2 尚未读取可信测试结果进行调参；性能基线文件可重建。

### Phase V2-1：P2 最小实现

任务：

1. 实现 UtilityNet；
2. 实现 Reliability-Constrained Gate；
3. 输出 utility/reliability/weight/system reliability；
4. 实现 ranking loss；
5. 实现验证集拒绝阈值；
6. 加入合成和小型真实 fixture 测试。

验收：

- 不可用权重精确为 0；
- 可用权重和为 1；
- 降低某模态可靠度且固定 utility 时，其权重不增加；
- `beta > 0`；
- ranking loss 手算一致；
- 推理删除故障元数据仍可运行；
- checkpoint 重载输出一致。

### Phase V2-2：扩展评估套件

任务：

1. 63-mask；
2. 混合故障；
3. 持续故障 episode；
4. Mask-only；
5. reliability calibration；
6. risk-coverage；
7. 用户级 Bootstrap。

验收：所有模型共享同一场景样本哈希；重复生成字节稳定；任何空指标保留 NaN；所有图表可追溯到 run ID。

### Phase V2-3：seed 13 五折与正式消融

任务：

1. 冻结 V2 seed 13 协议；
2. P2 五折；
3. P2-A1–A4 五折；
4. 生成中期报告；
5. 只基于训练/验证和预先规则决定是否进入 full。

验收：25 个计划单元（P2 五折 + 4×5 消融）状态完整；失败和重试日志保留；测试结果不反向修改协议。

### Phase V2-4：三种子核心矩阵

任务：

1. B4/B5/P 补 seed 29、47；
2. P2 运行 seed 29、47；
3. 校验 V1 seed 13 复用合同；
4. 生成三种子聚合和 Bootstrap 区间；
5. 运行选择性预测和扩展压力测试。

验收：B4/B5/P/P2 共 60 个核心 `fold × seed × model` 单元完整；所有场景源样本一致；聚合顺序符合第7节。

### Phase V2-5：Demo 2.0

页面必须展示：

- B4/B5/P/P2 同样本对比；
- 五维质量、可靠度、utility、最终权重；
- 系统可靠度和接受/拒绝状态；
- 模态组合选择；
- 混合故障与持续故障时间轴；
- 63-mask 聚合背景；
- 当前 checkpoint/run/fold/seed；
- 离线预计算特征回放声明。

验收：全部数字来自真实检查点或真实报告；合同不匹配时失败关闭；不存在随机预测回退。

### Phase V2-6：面试材料与发布审查

交付：

- README V2；
- V2 技术报告；
- V2 Model Card；
- V2 实验协议；
- 性能优化案例；
- 两分钟和八分钟讲稿；
- 面试追问库；
- 30–60 秒 Demo 录屏/GIF；
- 发布审查与来源映射。

不制作 PPT。公开发布前必须选择许可证并填写真实作者信息。

---

## 10. 测试清单

### 10.1 单元测试

- utility shape 与顺序；
- reliability 范围 `[0,1]`；
- `beta` 始终为正；
- fixed utility 下 reliability 降低不会增加权重；
- masked softmax 不变量；
- ranking loss 手算；
- system reliability 手算；
- abstention threshold 只读取 validation；
- 63 个非空 mask 唯一且完整；
- mixed failure 不污染已 drop 模态；
- episode 不跨用户或时间断点；
- Mask-only 不读取传感器特征；
- Bootstrap 以用户为单位且可复现；
- NaN 标签不进入任何损失/风险指标。

### 10.2 集成测试

- 合成多用户数据完成 P2 train/evaluate/report；
- P2 checkpoint/preprocessor/threshold/schema 合同重载；
- B4/B5/P/P2 在相同 63-mask 样本运行；
- mixed 与 persistent 场景生成全部预期文件；
- risk-coverage 阈值由 validation 保存并在 test 只读；
- Streamlit 无 UI 推理返回 utility/reliability/weight/decision。

### 10.3 科研有效性检查

- 打乱标签后性能下降；
- 置换 reliability target 后故障检测能力下降；
- `beta=0` 时 P2 退化为 utility-only 形式；
- 人工将 reliability 固定为1时结果符合预期基线；
- 测试标签文件不可被 threshold/calibration 代码打开；
- 所有模型受控场景样本 ID 哈希一致；
- Mask-only 与完整模型输入边界可审计；
- 失败 run 不进入最终聚合。

---

## 11. Coding Agent 工作规则

Agent 每次只执行一个 V2 Phase。开始前必须读取：

1. 本文件；
2. `docs/PROJECT_SPEC.md`；
3. `docs/EXPERIMENT_PROTOCOL.md`；
4. 当前测试、配置和 V1 协议锁。

每阶段结束必须报告：

1. 改动文件；
2. 设计选择；
3. 实际命令；
4. 测试结果；
5. run/报告产物；
6. 与验收条款的逐项映射；
7. 未解决问题和下一阶段前置条件。

不可变规则：

- 不覆盖 V1 run 和报告；
- 不在测试结果可见后无记录修改配置；
- 不把故障元数据作为推理输入；
- 不按行随机划分用户；
- 不把未知标签变成0；
- 不用测试集调阈值、校准或拒绝策略；
- 不删除失败实验；
- 不手填结果数字；
- 不跳过困难 mask/fold/seed；
- 新行为必须有测试；
- 影响冻结协议的修改必须写 ADR 并创建新版本。

### 11.1 可直接复制的总 Prompt

```text
你正在实现 RobustSense V2。先完整阅读 docs/RobustSense_V2_可执行项目总规约.md、V1 项目规约、实验协议、配置和测试，然后只执行我指定的 V2 Phase。

不可变约束：
1. 保护 V1 的 run、报告、协议锁和结果；V2 使用独立命名空间。
2. 任务仍是用户独立、多标签、未知标签掩码评估。
3. 训练统计和类别权重只来自 train；早停、分类阈值、校准和拒绝阈值只来自 validation；test 只评估。
4. P2 必须区分 quality、reliability、utility 和 fusion weight，不得把四者混为一谈。
5. 不可用模态权重为0，可用权重和为1，所有模态不可用必须报错或拒绝。
6. 故障类型和强度只能用于训练监督/评估元数据，不能成为推理输入。
7. 自然缺失、受控单点故障和 V2 扩展压力测试分开报告。
8. 所有数字来自真实 run；结果不理想也不得调换 seed、删场景或手改报告。

执行流程：检查仓库和 V1 哈希 → 写短计划与验收映射 → 最小实现 → 单元测试 → 集成测试 → 本阶段验收 → 报告产物与剩余风险。不要提前执行后续 Phase。
```

---

## 12. 建议开发节奏

### 12.1 五周完整版

| 周 | 任务 | 可展示里程碑 |
|---|---|---|
| 第1周 | V2-0、性能剖析、V1保护 | 性能基线、V2协议锁、无回归证明 |
| 第2周 | V2-1 | P2结构、可靠度约束和拒绝机制 |
| 第3周 | V2-2、V2-3 | 63-mask、混合/持续故障、seed13五折消融 |
| 第4周 | V2-4 | 三种子核心矩阵、置信区间、最终结果 |
| 第5周 | V2-5、V2-6 | Demo 2.0、README、报告和面试材料 |

### 12.2 两周面试压缩版

- 第1–2天：V1保护、性能基线、V2协议锁；
- 第3–5天：P2、ranking loss、选择性预测和测试；
- 第6–8天：63-mask、校准、risk-coverage；
- 第9–11天：P2五折seed13和关键消融；
- 第12–13天：Demo 2.0；
- 第14天：README、讲稿和问题库。

压缩版可以暂缓三种子、持续故障和扩展消融，但不能省略用户划分、防泄漏、真实检查点、五折 seed13 和结果可追溯性。

---

## 13. 风险清单

| 风险 | 症状 | 应对 |
|---|---|---|
| P2 权重被 reliability 完全支配 | utility 失去作用 | 监控 beta、做 utility-only/reliability-only 消融 |
| reliability 只记住人工强度 | 合成故障很好、自然缺失无效 | 分开报告自然与人工，增加 shortcut 诊断 |
| 拒绝机制靠测试调阈值 | test 风险异常漂亮 | 阈值只从 validation 保存，代码阻止 test 选择 |
| 63-mask 计算过慢 | 评估远慢于训练 | 批量 mask、共享前处理、缓存完整子集索引 |
| 持续 episode 跨时间断点 | 恢复指标失真 | 用户和时间差双重边界检查 |
| 多种子运行耗时过长 | GPU等待CPU、频繁I/O | 先完成性能剖析和向量化，再启动 full |
| 复用 V1 结果不公平 | 代码或配置已变化 | 合同哈希验证，不一致则重新运行对应基线 |
| 大量指标导致选择性汇报 | 只展示有利曲线 | 冻结主要终点，完整结果表全部保留 |
| Mask-only 表现较高 | 模型利用缺失模式捷径 | 置换、去 availability、分层结果和限制说明 |
| P2 不超过 P | 方法创新看似失败 | 分析校准/选择性/最坏场景，保留负结果 |
| 项目范围膨胀 | 同时做 Transformer、原始信号、端侧 | 严格 Phase gate，扩展仅在 V2 核心验收后 |

---

## 14. 面试展示设计

### 14.1 30秒版本

> RobustSense V2 研究传感器缺失或污染时的鲁棒多模态感知。我在60名用户、37.7万条记录的六模态数据上，按用户完成五折评估。V2 将传感器可靠度和任务效用分开建模，用可靠度约束门控抑制受损输入，并在整体输入不可信时拒绝预测。系统评估全部63种可用模态组合、混合和持续故障，所有结果可追溯到真实run和检查点。

V2 结果未完成前，口头表达必须把未来式与已完成事实分开。

### 14.2 面试代码讲解顺序

1. 数据与用户划分；
2. 标签 mask 和无泄漏预处理；
3. V1 B4/B5/P 的差异；
4. H4 负面结果；
5. P2 为什么解耦 reliability/utility；
6. 可靠度约束公式；
7. 63-mask 与选择性预测；
8. 多种子和用户级 Bootstrap；
9. 性能瓶颈和优化；
10. Demo 中一次真实故障回放。

### 14.3 必须会回答的问题

1. 为什么不直接让 Gate Weight 等于 Reliability？
2. `beta` 的作用是什么，如何防止为负？
3. 为什么拒绝阈值不能在测试集选？
4. 可靠度目标来自人工故障，会不会只适应模拟分布？
5. 63-mask 为什么比随机 drop-k 更完整？
6. 如何证明不同模型使用相同故障样本？
7. 为什么 Bootstrap 按用户而不是按行？
8. 多种子与多 fold 分别衡量什么变化？
9. 如果 P2 分类分数下降但校准改善，如何判断价值？
10. 低 GPU 利用率是否代表训练没有使用 GPU？
11. Mask-only 表现高说明什么？
12. 当前系统距离真实手机部署还差什么？

---

## 15. 最终交付清单

### 代码与协议

- [ ] V1 冻结与哈希保护；
- [ ] V2 配置和独立输出目录；
- [ ] P2 模型、损失、拒绝逻辑；
- [ ] 63-mask、mixed、persistent、Mask-only；
- [ ] 用户级 Bootstrap；
- [ ] V2 协议锁和 run registry；
- [ ] 全部新增行为测试通过。

### 实验

- [ ] P2 五折 seed13；
- [ ] P2-A1–A4 五折 seed13；
- [ ] B4/B5/P/P2 三种子核心矩阵；
- [ ] 自然缺失和 V1 受控场景；
- [ ] 63-mask；
- [ ] 混合故障；
- [ ] 持续故障；
- [ ] 可靠度校准和故障检测；
- [ ] Risk-Coverage；
- [ ] 用户级配对置信区间；
- [ ] 所有表图来源映射。

### 展示与面试

- [ ] Demo 2.0 使用真实检查点；
- [ ] 显示 q/r/u/w 与拒绝状态；
- [ ] 63-mask 热图与时间轴；
- [ ] README 架构图、结果和局限；
- [ ] 30–60秒演示录屏/GIF；
- [ ] 两分钟/八分钟讲稿；
- [ ] V2 面试问题库；
- [ ] 性能优化前后对比；
- [ ] LICENSE 和真实作者信息；
- [ ] 不制作 PPT。

---

## 16. 推荐 commit 顺序

```text
docs(v2): add executable v2 project specification
chore(v2): freeze v1 artifacts and create v2 namespaces
perf(train): add phase timing and throughput benchmark
perf(data): vectorize corruption and optimize data loading
feat(v2-model): add utility and reliability constrained fusion
feat(v2-train): add monotonic reliability ranking loss
feat(v2-eval): add reliability calibration and selective prediction
feat(v2-eval): add exhaustive 63-mask benchmark
feat(v2-eval): add mixed and persistent fault scenarios
feat(v2-eval): add mask-only shortcut diagnostics
feat(v2-stats): add user-clustered paired bootstrap
feat(v2-runner): add seed13 and full multi-seed registries
feat(app): add p2 diagnostics and abstention replay
docs(v2): add report model card interview guide and release review
```

每个 commit 都应在当时已有测试上通过。不得把全部 V2 代码一次性生成成不可审查的大提交。

---

## 17. 核心验收结论模板

项目结束时必须按以下模板填写，不得提前预填正向结果：

```text
V2 核心矩阵共计划 __ 个、成功 __ 个、失败 __ 个、重试 __ 个 run。
P2 相对 P 在自然缺失 Macro-F1 上变化 __，95% 用户级配对 Bootstrap 区间为 __。
在随机移除三个模态时，相对下降由 __ 变为 __。
63-mask 最坏 Macro-F1 由 __ 变为 __。
可靠度严重度排序正确率由 __ 变为 __。
在90%验证目标覆盖率下，测试覆盖率为 __，选择性风险为 __。
H1–H6 分别为：__。
未得到支持的假设及原因：__。
```

---

## 18. 第一条实际行动

本文件批准后，只执行 **Phase V2-0**：

1. 检查当前仓库状态；
2. 保存 V1 权威文件哈希和版本清单；
3. 建立 V2 目录、ADR、协议草案和性能基线脚本；
4. 运行 V1 回归，证明升级起点稳定；
5. 不实现 P2、不启动新训练、不读取新的测试结果用于决策。

Phase V2-0 验收后，才进入 P2 实现。本规约的目的不是拖慢升级，而是保证每一项新增结果都能在面试中被追问、被验证、被复现。
