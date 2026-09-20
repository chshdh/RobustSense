# Phase V2-5 验收报告：Demo 2.0

- 阶段：V2-5
- 状态：通过
- 日期：2026-09-19
- 协议锁：`reports/v2/phase_v2_5_protocol_lock.json`

## 1. 完成内容

Demo 2.0 已升级为完全由 V2-4 真实登记表、checkpoint 和报告驱动的离线回放页面。
页面支持选择五个官方 fold、三个正式 seed 和 B4/B5/P/P2 四个模型，并在同一测试
split、同一样本索引和同一故障参数下比较四模型。

页面包含：

1. B4/B5/P/P2 同样本预测对比；
2. 每模态五维质量、预测可靠度、P2 utility 和最终融合权重；
3. P2 系统可靠度、validation 拒绝阈值和接受/拒绝状态；
4. 模态关闭与 Gaussian、偏置、尺度混合故障；
5. 持续故障 `pre → fault → post` 分类时间轴和真实 episode 故障分数；
6. 63-mask 与混合故障五折、三种子聚合背景；
7. 当前 checkpoint、run、fold、seed、测试 split 哈希和协议哈希；
8. 离线预计算特征回放边界声明。

## 2. 失败关闭合同

run 路径只能从 `phase_v2_4_run_registry.csv` 解析。页面加载时校验 V2-4 协议锁、
60/60 核心和 60/60 扩展状态、checkpoint 合同、分类阈值来源、P2 拒绝阈值来源、
自然评估身份、扩展评估身份以及报告文件 SHA-256。任一条件不满足即停止页面。

页面没有随机预测、固定示例概率或合成替代值。样本按钮只按索引移动，不影响模型输出。
B4/B5 没有可靠度和 utility 时显示“不适用”，不会用 0 或伪数值填充。

## 3. 真实回放验收

自动验收使用 `fold 0 / seed 29` 的四个真实 run，在同一个完整测试样本上关闭
`phone_acc`，同时向 `phone_gyro` 注入 `Gaussian σ=2`：

| 模型 | run |
|---|---|
| B4 | `extrasensory-gated-fold0-seed29-full` |
| B5 | `extrasensory-robust-gated-fold0-seed29-full` |
| P | `extrasensory-quality-aware-fold0-seed29-full` |
| P2 | `extrasensory-p2-fold0-seed29-v2_multiseed` |

四模型测试 split SHA-256 均为
`25ef008914f920caecbbf385bad5f1ab99ba8faf385016333a7338fb4f57502a`。
被关闭模态权重严格为 0，所有可用模态最终权重和为 1。P2 的 utility 与可靠度均为有限
实数，系统可靠度、拒绝阈值和接受/拒绝由真实前向输出产生。

报告背景合同同时通过：252 行 mask 汇总、240 行混合故障汇总、12 行所选持续故障
三阶段汇总和 3 点真实 episode 时间轴。

## 4. 自动化验收

```text
Ruff                                      All checks passed
pytest                                    78 passed, 0 failed, 0 errors
pip check                                 No broken requirements found
V1 baseline manifest                      813/813 files verified
真实四模型混合故障回放                   passed
Streamlit AppTest                         0 exceptions，5 个功能页签
Streamlit /_stcore/health                 HTTP 200，ok
随机预测回退                             false
合同不匹配回退                           false
```

健康检查后已关闭验收服务器和 8519 端口，不在后台保留演示进程。

## 5. 使用方式

在项目根目录运行：

```powershell
.\scripts\run_demo.ps1
```

然后访问终端显示的本机地址。首次加载会校验并加载四个 checkpoint，因此比后续交互慢。

## 6. 权威产物

| 产物 | 路径 |
|---|---|
| Demo 页面 | `app/streamlit_app.py` |
| 失败关闭推理层 | `src/robustsense/v2_demo.py` |
| V2-5 协议 | `docs/v2/EXPERIMENT_PROTOCOL_V2_5.md` |
| V2-5 协议锁 | `reports/v2/phase_v2_5_protocol_lock.json` |
| 真实回放验收 | `reports/v2/phase_v2_5_acceptance.json` |
| 页面健康检查 | `reports/v2/phase_v2_5_streamlit_health.json` |
| pytest 机器报告 | `reports/v2/phase_v2_5_pytest.xml` |

## 7. 下一阶段

V2-5 阶段门已通过。下一阶段是 V2-6：更新 README、V2 技术报告、V2 Model Card、
性能优化案例、两分钟和八分钟讲稿、面试追问库、Demo 录屏说明和发布审查。按项目所有者
要求不制作 PPT；公开发布前仍需确定许可证并填写真实作者信息。
