# Phase V2-0 验收报告：冻结与保护

- 状态：通过
- 完成日期：2026-09-17
- 范围：只执行 V1 冻结、V2 隔离、协议锁、环境核验、性能基线和 V1 回归
- 未执行：P2 实现、新可信训练、新 test 结果生成或基于 test 的决策

## 1. 阶段产物

| 产物 | 路径 | 作用 |
|---|---|---|
| V1 冻结清单 | `reports/v2/v1_baseline_manifest.json` | 校验 813 个 V1 权威/身份文件 |
| V2-0 协议锁 | `reports/v2/phase_v2_0_protocol_lock.json` | 锁定保护规则、脚本与性能协议 |
| 环境报告 | `reports/v2/environment.json` | 记录 Python、依赖、CUDA 与 GPU |
| 性能基线 | `reports/v2/performance_baseline.json` | 记录训练各阶段耗时和吞吐 |
| 报告重建证据 | `reports/v2/v1_report_rebuild.json` | 记录 V1 双重重建哈希对比 |
| ADR | `docs/v2/ADR-0001-V1保护与V2隔离.md` | 解释 V1/V2 隔离决策 |
| 协议草案 | `docs/v2/EXPERIMENT_PROTOCOL_DRAFT.md` | 固定防泄漏与阶段边界 |

V2 已建立 `configs/v2/`、`docs/v2/`、`reports/v2/` 和 `runs/v2/`。P2、消融、
63-mask 与正式评估子目录目前只保留边界说明，没有伪造尚未实现的配置。

## 2. V1 冻结结果

- 文件数：813。
- 成功 run 身份数：66，其中正式可信 run 为 60，开发消融为 6。
- V1 清单内容哈希：
  `1c6278235f1687f962d426b134498d7884009db2005b386e6330304e3f23952f`。
- V1 Phase 5 协议哈希：
  `91a4e5fd7547ecf347d7fb3d7ee101d7389ac71f77386d4301e1bc1ec8929a7b`。
- V2-0 协议哈希：
  `fd5be4f32144e6186bf1c88912a8b5b76fdfdef6c908204a84bafd8501269c55`。
- 冻结后再次校验：813/813 一致，问题数 0。

仓库的 `.git` 已存在，但 `main` 尚无 commit，无法创建有意义的 V1 tag。因此本阶段以文件
哈希为权威版本证据，没有伪造 Git 历史。大型 Parquet 预测/诊断文件约占整个 `runs/` 的
主要体积，没有重复全量哈希；它们由冻结的检查点、run 身份文件、评估清单和权威聚合输出
约束，同时 V2 被禁止写入 V1 run 路径。

## 3. 环境核验

| 项目 | 实测值 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| CUDA 可用 | 是 |
| GPU | NVIDIA GeForce RTX 2060，6,442,123,264 bytes |
| Compute capability | 7.5 |
| GPU tensor smoke | 通过，求和结果 3.0 |

没有新建或复现虚拟环境，继续使用项目既有 `.venv-dev`。第一次环境导入曾被 Windows 应用
控制瞬时拦截 NumPy DLL，随后相同环境连续正常导入并完成全部核验；没有重装包或改依赖。

## 4. 性能基线

工作负载为 V1 P（quality-aware）、fold0、seed13、batch size 512、2 次 warm-up、10 次
测量。只读取 train 与 validation，不读取 test，不计算准确率，不保存检查点，也不创建 run。

| 阶段 | 平均耗时（ms） | P95（ms） |
|---|---:|---:|
| DataLoader 等待 | 10.13 | 13.62 |
| batch 搬运 | 3.56 | 6.03 |
| corruption | 269.90 | 291.05 |
| forward | 34.80 | 40.19 |
| backward / optimizer | 48.08 | 64.15 |
| validation | 27.13 | 31.86 |
| controlled evaluation | 36.24 | 41.80 |
| JSON 写入 | 4.09 | 4.09 |
| Parquet 写入 | 30.81 | 30.81 |

- 估算训练 step：366.47 ms。
- 估算纯训练吞吐：1,397.11 samples/s。
- 包含 validation 与 controlled evaluation 的实测吞吐：1,188.56 samples/s。
- CUDA 峰值已分配显存：34,057,728 bytes。
- 最大瓶颈：`corruption`，占估算训练 step 的 73.65%。

结论：模型 forward/backward 确实运行在 GPU；低 GPU 利用率的主要原因是
`CorruptionRegistry._apply` 在 CPU 上逐样本执行 Python 循环和随机采样，GPU 在每个 batch
之前等待约 270 ms。下一阶段的安全提速方向是先为污染等价性建立测试，再向量化污染、减少
逐样本 Python 调度，并评估 DataLoader 预取；不能以牺牲样本级确定性为代价直接改写。

## 5. 回归证据

| 检查 | 结果 |
|---|---|
| 原 V1 测试 | 37 passed |
| V2-0 新增哈希测试 | 2 passed |
| 最终完整测试 | 39 passed in 52.74s |
| Ruff | All checks passed |
| `pip check` | No broken requirements found |
| V1 报告第一次重建 | 31 个产物，变化 0 |
| V1 报告第二次重建 | 31 个产物，变化 0 |
| V1 冻结清单复核 | 813/813，通过 |

pytest 默认临时目录属于另一个沙箱账户，最初导致 14 个 fixture 无权限；改为项目内临时目录
后 V1 37 项全部通过。最终一次完整测试又因临时路径前缀过长触发 Windows 路径长度限制，改用
短路径后 39 项全部通过。这两项均属于执行容器路径约束，不是产品代码失败。

## 6. 验收结论

Phase V2-0 的六项任务全部完成：V1 权威产物已冻结、V2 命名空间已隔离、ADR 与协议已锁、
CUDA 环境已核验、性能瓶颈已量化、V1 测试和报告重建均无回归。项目现在具备进入
Phase V2-1（P2 最小实现）的稳定起点。

