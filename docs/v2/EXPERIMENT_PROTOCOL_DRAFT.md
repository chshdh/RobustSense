# RobustSense V2 实验协议草案

## 1. 当前协议级别

本文档在 Phase V2-0 锁定升级实验的边界、数据隔离和验收方法，不锁定尚未实现的 P2
超参数。P2、63-mask、混合/持续故障和三种子实验将在对应阶段通过版本化配置与新协议锁
补充，禁止回填或覆盖 V2-0 锁。

## 2. 不可变任务定义

- 数据集仍为 ExtraSensory，60 名用户、377,346 行、177 个特征、6 个模态、15 个标签。
- 划分单位是用户；同一用户不得跨 train、validation、test。
- 未知标签必须通过 `target_mask` 排除，不能当作负类。
- 中位数、IQR、质量统计量和类别权重只由 train 拟合。
- 早停、分类阈值、校准与拒绝阈值只由 validation 选择。
- test 只用于冻结方案的最终评估，不能用于选模型、调参或选择阈值。

## 3. V1 保护协议

- V1 权威锚点：`reports/v2/v1_baseline_manifest.json`。
- V1 Phase 5 协议哈希：
  `91a4e5fd7547ecf347d7fb3d7ee101d7389ac71f77386d4301e1bc1ec8929a7b`。
- V1 可信矩阵必须保持 60/60，开发消融保持独立，不冒充正式五折结果。
- V2 不写入既有 `runs/extrasensory-*`，不覆盖 `reports/` 下的 V1 权威文件。
- 共享代码变更后，先执行 V1 清单验证、测试和报告重建，再执行 V2 实验。

## 4. V2 命名空间

| 类型 | 路径 |
|---|---|
| 配置 | `configs/v2/` |
| 文档与 ADR | `docs/v2/` |
| 报告与锁 | `reports/v2/` |
| 运行产物 | `runs/v2/` |

V2 run ID 必须含模型、fold、seed 和 profile，且不能与 V1 run ID 重名。

## 5. Phase V2-0 性能基线

基线使用 V1 的 quality-aware 模型、fold0、seed13、batch size 512。只读取 train 和
validation，执行 2 次 warm-up 与 10 次短程测量；模型和优化器仅存在于内存，不保存检查点，
不形成正式训练 run。

必须分别测量：DataLoader 等待、batch 搬运、训练污染、forward、backward/optimizer、
validation、controlled evaluation、JSON 写入和 Parquet 写入。GPU 异步操作前后必须同步，
否则计时无效。性能基线不计算准确率，不读取 test，不用于宣称模型效果。

## 6. Phase V2-0 验收

1. V1 清单可重复校验，60/60 可信 run 状态不变。
2. 既有 37 项 V1 测试通过；V2-0 新增测试也通过。
3. V1 报告连续重建两次，权威输出 SHA-256 变化数为 0。
4. CUDA、GPU 和依赖核验通过。
5. 性能基线可由脚本重建，且明确记录未读取 test、未产生正式 run。
6. P2 尚未实现，没有启动新的可信训练，也没有使用新的 test 结果做决策。

