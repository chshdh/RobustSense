# V2 故障配置

- `train.yaml`：P2 开发训练污染配置；
- `exhaustive_masks.yaml`：63 个非空 availability mask；
- `mixed_failures.yaml`：60 个 drop + Gaussian 混合故障；
- `persistent_failures.yaml`：持续故障 episode 的时间与长度合同。

训练增强与扩展评估仍是两套独立配置，不能把训练时随机污染冒充测试场景。
