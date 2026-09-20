"""RobustSense V2-5：真实 checkpoint 驱动的离线回放 Demo 2.0。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from robustsense.constants import MODALITIES
from robustsense.v2_demo import (
    DEMO_FOLDS,
    DEMO_MODELS,
    DEMO_SEEDS,
    QUALITY_COLUMNS,
    V2DemoArtifactError,
    infer_sample,
    load_demo_matrix,
    mask_background,
    mixed_background,
    persistent_episode_timeline,
    persistent_phase_timeline,
    validate_v2_demo_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {
    "B4": "B4 · 普通门控融合",
    "B5": "B5 · 鲁棒门控融合",
    "P": "P · 质量感知融合",
    "P2": "P2 · 可靠度约束融合",
}
MODALITY_LABELS = {
    "phone_acc": "手机加速度",
    "phone_gyro": "手机陀螺仪",
    "watch_acc": "手表加速度",
    "location": "位置",
    "audio": "音频",
    "phone_state": "手机状态",
}
PHASE_LABELS = {"pre": "故障前", "fault": "故障中", "post": "故障后"}


@st.cache_resource(show_spinner="正在校验登记表并加载四个真实 checkpoint……")
def cached_matrix(fold: int, seed: int, device_name: str):
    return load_demo_matrix(
        PROJECT_ROOT, fold, seed, device_name=device_name
    )


@st.cache_data(show_spinner=False)
def cached_context() -> dict:
    context = validate_v2_demo_context(PROJECT_ROOT)
    return {
        "protocol_sha256": context["protocol_sha256"],
        "bootstrap": context["bootstrap"],
    }


@st.cache_data(show_spinner=False)
def cached_mask_background() -> pd.DataFrame:
    return mask_background(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_mixed_background() -> pd.DataFrame:
    return mixed_background(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_persistent_timeline(
    fault: str, target_modality: str, fault_length: int
) -> pd.DataFrame:
    return persistent_phase_timeline(
        PROJECT_ROOT,
        fault=fault,
        target_modality=target_modality,
        fault_length=fault_length,
    )


def severity_control(fault_type: str) -> float:
    if fault_type == "gaussian":
        return float(
            st.sidebar.select_slider(
                "Gaussian σ", options=[0.25, 0.5, 1.0, 2.0], value=1.0
            )
        )
    if fault_type == "bias":
        return float(
            st.sidebar.select_slider(
                "标准化偏置", options=[0.25, 0.5, 1.0], value=0.5
            )
        )
    if fault_type == "scale":
        return float(
            st.sidebar.select_slider(
                "缩放倍数", options=[0.5, 0.75, 1.25, 1.5], value=1.25
            )
        )
    return 0.0


def prediction_frame(clean: dict, current: dict) -> pd.DataFrame:
    targets = [
        int(value) if known else "未知"
        for value, known in zip(
            current["targets"], current["target_mask"], strict=True
        )
    ]
    return (
        pd.DataFrame(
            {
                "标签": current["labels"],
                "自然概率": clean["probabilities"],
                "当前概率": current["probabilities"],
                "变化": current["probabilities"] - clean["probabilities"],
                "验证集阈值": current["thresholds"],
                "当前预测": current["predictions"],
                "真实标签": targets,
            }
        )
        .sort_values("当前概率", ascending=False)
        .reset_index(drop=True)
    )


def modality_frame(current: dict) -> pd.DataFrame:
    quality = pd.DataFrame(
        current["quality_features"],
        columns=[f"质量_{name}" for name in QUALITY_COLUMNS],
    )
    quality.insert(0, "模态", [MODALITY_LABELS[name] for name in current["modalities"]])
    quality.insert(1, "实际可用", current["availability"])
    quality["预测可靠度"] = current["reliability"]
    quality["任务 utility"] = current["utility_scores"]
    quality["最终权重"] = current["fusion_weights"]
    return quality


def comparison_frame(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_id in DEMO_MODELS:
        result = results[model_id]
        rows.append(
            {
                "模型": MODEL_LABELS[model_id],
                "run": result["run_id"],
                "阳性标签数": int(result["predictions"].sum()),
                "平均概率": float(np.mean(result["probabilities"])),
                "最高概率标签": result["labels"][int(np.argmax(result["probabilities"]))],
                "最高概率": float(np.max(result["probabilities"])),
                "系统可靠度": result["system_reliability"],
                "接受/拒绝": result["acceptance"],
            }
        )
    return pd.DataFrame(rows)


def render_single_sample(
    primary_model: str,
    clean: dict,
    current: dict,
    all_current: dict[str, dict],
) -> None:
    st.subheader("四模型同样本、同故障对比")
    st.dataframe(
        comparison_frame(all_current).style.format(
            {
                "平均概率": "{:.4f}",
                "最高概率": "{:.4f}",
                "系统可靠度": "{:.4f}",
            },
            na_rep="不适用",
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "四行使用完全相同的测试样本、模态组合和故障参数；P2 的接受/拒绝由 validation "
        "冻结阈值决定。"
    )

    predictions = prediction_frame(clean, current)
    chart_rows = predictions.head(10).melt(
        id_vars="标签",
        value_vars=["自然概率", "当前概率"],
        var_name="视图",
        value_name="概率",
    )
    figure = px.bar(
        chart_rows,
        x="概率",
        y="标签",
        color="视图",
        barmode="group",
        orientation="h",
        range_x=[0, 1],
        title=f"{MODEL_LABELS[primary_model]}：自然输入与当前故障输入",
    )
    figure.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        predictions.style.format(
            {
                "自然概率": "{:.4f}",
                "当前概率": "{:.4f}",
                "变化": "{:+.4f}",
                "验证集阈值": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_quality(current: dict) -> None:
    st.subheader("五维质量、可靠度、utility 与最终权重")
    frame = modality_frame(current)
    st.dataframe(
        frame.style.format(
            {
                "质量_availability": "{:.1f}",
                "质量_observed_fraction": "{:.3f}",
                "质量_outlier_fraction": "{:.3f}",
                "质量_mean_abs_robust_z": "{:.3f}",
                "质量_max_abs_robust_z": "{:.3f}",
                "预测可靠度": "{:.4f}",
                "任务 utility": "{:.4f}",
                "最终权重": "{:.4f}",
            },
            na_rep="不适用",
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "utility 只由 P2 独立输出；P 的门控分数没有被伪装成 utility。不可用模态权重必须"
        "严格为 0，可用模态权重之和必须为 1。"
    )


def render_persistent(bundles: dict, primary_model: str) -> None:
    st.subheader("持续故障三阶段时间轴")
    col1, col2, col3 = st.columns(3)
    fault = col1.selectbox(
        "持续故障类型",
        options=["drop", "gaussian_sigma_2"],
        format_func=lambda value: "模态掉线" if value == "drop" else "Gaussian σ=2",
    )
    target = col2.selectbox(
        "持续故障目标模态",
        options=list(MODALITIES),
        format_func=lambda value: MODALITY_LABELS[value],
    )
    length = int(col3.selectbox("故障长度（步）", options=[5, 15, 30], index=1))
    phase = cached_persistent_timeline(fault, target, length).copy()
    phase["阶段"] = phase["phase"].map(PHASE_LABELS)
    phase["模型"] = phase["model_id"].map(MODEL_LABELS)
    phase_chart = px.line(
        phase,
        x="阶段",
        y="macro_f1",
        color="模型",
        markers=True,
        category_orders={"阶段": ["故障前", "故障中", "故障后"]},
        labels={"macro_f1": "Macro-F1"},
        title="真实扩展报告：故障前 → 故障中 → 故障后",
    )
    st.plotly_chart(phase_chart, use_container_width=True)

    reliability_model = primary_model if primary_model in {"P", "P2"} else "P2"
    try:
        episode, metadata = persistent_episode_timeline(
            bundles[reliability_model],
            fault=fault,
            target_modality=target,
            fault_length=length,
        )
    except V2DemoArtifactError as exc:
        st.info(str(exc))
        return
    episode["阶段"] = episode["phase"].map(PHASE_LABELS)
    response_chart = px.line(
        episode,
        x="阶段",
        y="fault_score",
        markers=True,
        category_orders={"阶段": ["故障前", "故障中", "故障后"]},
        labels={"fault_score": "目标模态故障分数"},
        title=f"{MODEL_LABELS[reliability_model]}：真实 episode 可靠度响应",
    )
    st.plotly_chart(response_chart, use_container_width=True)
    st.caption(
        f"episode={metadata['episode_id']} · 检测={metadata['detected']} · "
        f"恢复={metadata['recovered']} · 检测延迟={metadata['detection_delay_steps']} 步 · "
        f"恢复步数={metadata['recovery_steps']}"
    )


def render_aggregate_context() -> None:
    st.subheader("63-mask 与混合故障聚合背景")
    masks = cached_mask_background()
    mask_curve = (
        masks.groupby(["model_id", "available_count"], as_index=False)["macro_f1"]
        .mean()
        .assign(模型=lambda frame: frame["model_id"].map(MODEL_LABELS))
    )
    mask_figure = px.line(
        mask_curve,
        x="available_count",
        y="macro_f1",
        color="模型",
        markers=True,
        labels={"available_count": "可用模态数", "macro_f1": "Macro-F1"},
        title="63 个非空模态组合：按可用模态数汇总",
    )
    st.plotly_chart(mask_figure, use_container_width=True)

    mixed = cached_mixed_background()
    mixed_curve = (
        mixed.groupby(["model_id", "gaussian_sigma"], as_index=False)["macro_f1"]
        .mean()
        .assign(模型=lambda frame: frame["model_id"].map(MODEL_LABELS))
    )
    mixed_figure = px.bar(
        mixed_curve,
        x="gaussian_sigma",
        y="macro_f1",
        color="模型",
        barmode="group",
        labels={"gaussian_sigma": "Gaussian σ", "macro_f1": "Macro-F1"},
        title="混合故障：丢弃一个模态，同时污染另一个模态",
    )
    st.plotly_chart(mixed_figure, use_container_width=True)
    st.caption("图中数字来自 V2-4 五折、三种子聚合报告，不是当前单样本推理的估计值。")


def main() -> None:
    st.set_page_config(page_title="RobustSense Demo 2.0", page_icon="🛡️", layout="wide")
    st.title("RobustSense · 可靠度约束多模态感知 Demo 2.0")
    st.warning(
        "本页面只回放 ExtraSensory 离线预计算测试特征。它不是实时传感器采集、原始信号"
        "处理、手机端部署或物理故障模拟器。"
    )
    try:
        context = cached_context()
    except (V2DemoArtifactError, FileNotFoundError, ValueError) as exc:
        st.error(f"V2-4 权威产物校验失败，Demo 已关闭：{exc}")
        st.stop()

    fold = int(st.sidebar.selectbox("官方测试 fold", options=list(DEMO_FOLDS), index=0))
    seed = int(st.sidebar.selectbox("随机种子", options=list(DEMO_SEEDS), index=1))
    primary_model = st.sidebar.selectbox(
        "主模型", options=list(DEMO_MODELS), index=3, format_func=MODEL_LABELS.get
    )
    device_name = st.sidebar.selectbox(
        "推理设备",
        options=["auto", "cpu"],
        format_func=lambda value: "自动（优先 GPU）" if value == "auto" else "CPU",
    )
    try:
        bundles = cached_matrix(fold, seed, device_name)
    except (V2DemoArtifactError, FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"真实 run 合同校验失败，Demo 已关闭：{exc}")
        st.stop()
    primary = bundles[primary_model]

    sample_key = f"v2-sample-{fold}-{seed}"
    if sample_key not in st.session_state:
        st.session_state[sample_key] = 0
    if st.sidebar.button("下一条测试样本"):
        st.session_state[sample_key] = (
            int(st.session_state[sample_key]) + 1
        ) % len(primary.dataset)
    sample_index = int(
        st.sidebar.number_input(
            "测试样本索引",
            min_value=0,
            max_value=len(primary.dataset) - 1,
            value=int(st.session_state[sample_key]),
            step=1,
        )
    )
    st.session_state[sample_key] = sample_index
    natural_availability = primary.dataset[sample_index]["availability"]
    naturally_available = [
        name
        for name, available in zip(MODALITIES, natural_availability, strict=True)
        if bool(available)
    ]
    disabled = tuple(
        st.sidebar.multiselect(
            "关闭当前可用模态",
            options=naturally_available,
            format_func=lambda value: MODALITY_LABELS[value],
        )
    )
    fault_label = st.sidebar.selectbox(
        "特征空间故障",
        options=["none", "gaussian", "bias", "scale"],
        format_func={
            "none": "无额外污染",
            "gaussian": "Gaussian 噪声",
            "bias": "偏置漂移",
            "scale": "尺度误差",
        }.get,
    )
    fault_type = "clean" if fault_label == "none" else fault_label
    target_options = [name for name in naturally_available if name not in disabled]
    if not target_options:
        st.error("至少必须保留一个自然可用模态。")
        st.stop()
    target_modality = (
        st.sidebar.selectbox(
            "污染目标模态",
            options=target_options,
            format_func=lambda value: MODALITY_LABELS[value],
        )
        if fault_type != "clean"
        else None
    )
    severity = severity_control(fault_type)

    try:
        clean = infer_sample(primary, sample_index)
        all_current = {
            model_id: infer_sample(
                bundle,
                sample_index,
                disabled_modalities=disabled,
                fault_type=fault_type,
                target_modality=target_modality,
                severity=severity,
            )
            for model_id, bundle in bundles.items()
        }
    except (V2DemoArtifactError, RuntimeError, ValueError, IndexError) as exc:
        st.error(f"推理合同不满足，Demo 已关闭：{exc}")
        st.stop()
    current = all_current[primary_model]

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("主模型", primary_model)
    top2.metric("测试样本", f"{sample_index:,} / {len(primary.dataset) - 1:,}")
    top3.metric(
        "系统可靠度",
        "不适用"
        if current["system_reliability"] is None
        else f"{current['system_reliability']:.4f}",
    )
    top4.metric("接受/拒绝", current["acceptance"])
    st.caption(
        f"run={current['run_id']} · fold={fold} · seed={seed} · device={current['device']} · "
        f"用户哈希={current['user']} · timestamp={current['timestamp']}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["同样本对比", "质量与门控", "持续故障时间轴", "聚合背景", "产物合同"]
    )
    with tab1:
        render_single_sample(primary_model, clean, current, all_current)
    with tab2:
        render_quality(current)
    with tab3:
        render_persistent(bundles, primary_model)
    with tab4:
        render_aggregate_context()
    with tab5:
        st.subheader("可追溯产物")
        artifact_rows = [
            {
                "模型": model_id,
                "run": bundle.run_id,
                "checkpoint": str(bundle.checkpoint_path),
                "源阶段": bundle.source_phase,
                "fold": bundle.fold,
                "seed": bundle.seed,
                "测试 split SHA-256": bundle.test_split_sha256,
            }
            for model_id, bundle in bundles.items()
        ]
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
        bootstrap = context["bootstrap"]
        st.markdown(
            f"""
- V2-4 协议哈希：`{context['protocol_sha256']}`
- 当前主 checkpoint：`{current['checkpoint']}`
- 分类阈值来源：validation
- P2 拒绝阈值：`{current['abstention_threshold']}`
- 自然缺失 P2-P Bootstrap：均值 `{bootstrap['mean_delta_p2_minus_p']:.6f}`，
  95% 区间 `[{bootstrap['ci_2_5']:.6f}, {bootstrap['ci_97_5']:.6f}]`
- 页面没有随机预测或合成概率回退；任何合同不匹配都会停止运行。
"""
        )


if __name__ == "__main__":
    main()
