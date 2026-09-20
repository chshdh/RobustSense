"""Phase V2-2 calibration, selective-risk, episode, and bootstrap metrics."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from robustsense.training.phase2_metrics import phase2_multilabel_metrics


def _nan_if_none(value: Any) -> Any:
    return np.nan if value is None else value


def reliability_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    *,
    modality_names: list[str],
    n_bins: int,
    run_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Build fixed-bin reliability calibration tables without filling empty bins."""
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != expected.shape or expected.shape != valid.shape:
        raise ValueError("Reliability predictions, targets, and masks must align")
    if predicted.ndim != 2 or predicted.shape[1] != len(modality_names):
        raise ValueError("Modality names do not match reliability width")
    if n_bins <= 0 or not run_id:
        raise ValueError("n_bins and run_id must be non-empty")
    finite_valid = valid & np.isfinite(predicted) & np.isfinite(expected)
    if ((predicted[finite_valid] < 0) | (predicted[finite_valid] > 1)).any():
        raise ValueError("Reliability predictions must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for modality_index, modality in enumerate(modality_names):
        modality_valid = finite_valid[:, modality_index]
        values = predicted[modality_valid, modality_index]
        truth = expected[modality_valid, modality_index]
        weighted_gap = 0.0
        for bin_index in range(n_bins):
            lower = float(edges[bin_index])
            upper = float(edges[bin_index + 1])
            selected = (values >= lower) & (
                values <= upper if bin_index == n_bins - 1 else values < upper
            )
            count = int(selected.sum())
            mean_prediction = float(values[selected].mean()) if count else np.nan
            mean_target = float(truth[selected].mean()) if count else np.nan
            gap = abs(mean_prediction - mean_target) if count else np.nan
            if count:
                weighted_gap += count * gap
            bins.append(
                {
                    "run_id": run_id,
                    "modality": modality,
                    "bin_index": bin_index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": count,
                    "mean_prediction": mean_prediction,
                    "mean_target": mean_target,
                    "absolute_gap": gap,
                }
            )
        count = int(modality_valid.sum())
        summary.append(
            {
                "run_id": run_id,
                "modality": modality,
                "valid_count": count,
                "expected_calibration_error": weighted_gap / count if count else np.nan,
            }
        )
    return {"bins": bins, "summary": summary}


def fault_detection_metrics(
    reliability: np.ndarray,
    artificial_fault: np.ndarray,
    valid_mask: np.ndarray,
    *,
    run_id: str,
) -> dict[str, float | int | str]:
    predicted = np.asarray(reliability, dtype=np.float64)
    fault = np.asarray(artificial_fault, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != fault.shape or fault.shape != valid.shape:
        raise ValueError("Reliability, fault labels, and valid masks must align")
    selected = valid & np.isfinite(predicted)
    labels = fault[selected].astype(np.int64)
    scores = 1.0 - predicted[selected]
    has_two_classes = len(np.unique(labels)) == 2
    return {
        "run_id": run_id,
        "valid_count": int(selected.sum()),
        "positive_count": int(labels.sum()),
        "fault_auroc": float(roc_auc_score(labels, scores)) if has_two_classes else np.nan,
        "fault_auprc": (
            float(average_precision_score(labels, scores)) if has_two_classes else np.nan
        ),
    }


def masked_binary_cross_entropy_per_sample(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    if predicted.shape != expected.shape or expected.shape != known.shape:
        raise ValueError("Probabilities, targets, and masks must align")
    clipped = np.clip(predicted, 1.0e-7, 1.0 - 1.0e-7)
    clean_targets = np.nan_to_num(expected, nan=0.0)
    losses = -(clean_targets * np.log(clipped) + (1.0 - clean_targets) * np.log(1 - clipped))
    counts = known.sum(axis=1)
    output = np.full(len(predicted), np.nan, dtype=np.float64)
    usable = counts > 0
    output[usable] = (losses * known).sum(axis=1)[usable] / counts[usable]
    return output


def risk_coverage_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    system_reliability: np.ndarray,
    *,
    labels: list[str],
    thresholds: np.ndarray | float,
    coverage_grid: np.ndarray,
    run_id: str,
) -> list[dict[str, Any]]:
    predicted = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    reliability = np.asarray(system_reliability, dtype=np.float64)
    coverage = np.asarray(coverage_grid, dtype=np.float64)
    if predicted.shape != expected.shape or expected.shape != known.shape:
        raise ValueError("Probabilities, targets, and masks must align")
    if reliability.shape != (len(predicted),):
        raise ValueError("System reliability must have one value per sample")
    if ((coverage < 0) | (coverage > 1)).any() or not run_id:
        raise ValueError("Coverage must be in [0, 1] and run_id must be non-empty")
    order = np.argsort(-reliability, kind="stable")
    losses = masked_binary_cross_entropy_per_sample(predicted, expected, known)
    rows: list[dict[str, Any]] = []
    for requested in coverage:
        accepted_count = int(np.ceil(float(requested) * len(predicted)))
        accepted = order[:accepted_count]
        finite_losses = losses[accepted][np.isfinite(losses[accepted])]
        metrics: dict[str, Any] = {}
        if accepted_count:
            metrics = phase2_multilabel_metrics(
                predicted[accepted], expected[accepted], known[accepted], labels, thresholds
            )
        rows.append(
            {
                "run_id": run_id,
                "requested_coverage": float(requested),
                "coverage": accepted_count / len(predicted) if len(predicted) else np.nan,
                "accepted_count": accepted_count,
                "risk_masked_bce": (
                    float(finite_losses.mean()) if len(finite_losses) else np.nan
                ),
                "macro_f1": _nan_if_none(metrics.get("macro_f1")),
                "micro_f1": _nan_if_none(metrics.get("micro_f1")),
                "known_target_count": int(metrics.get("known_target_count", 0)),
            }
        )
    return rows


def persistent_episode_response(
    fault_scores: np.ndarray,
    *,
    pre_indices: tuple[int, ...],
    fault_indices: tuple[int, ...],
    post_indices: tuple[int, ...],
    detection_threshold: float,
    episode_id: str,
    run_id: str,
    recovery_tolerance: float = 0.05,
) -> dict[str, Any]:
    scores = np.asarray(fault_scores, dtype=np.float64)
    if not 0 <= detection_threshold <= 1 or recovery_tolerance < 0:
        raise ValueError("Detection threshold and recovery tolerance are invalid")
    phases = {"pre": pre_indices, "fault": fault_indices, "post": post_indices}
    means = {
        f"{phase}_mean_fault_score": (
            float(np.nanmean(scores[list(indices)])) if indices else np.nan
        )
        for phase, indices in phases.items()
    }
    detected = [
        offset
        for offset, index in enumerate(fault_indices)
        if scores[index] >= detection_threshold
    ]
    pre_values = scores[list(pre_indices)]
    finite_pre = pre_values[np.isfinite(pre_values)]
    recovery_upper = (
        min(1.0, float(finite_pre.mean()) + recovery_tolerance)
        if len(finite_pre)
        else np.nan
    )
    recovered = [
        offset
        for offset, index in enumerate(post_indices)
        if np.isfinite(recovery_upper) and scores[index] <= recovery_upper
    ]
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        **means,
        "detection_delay_steps": float(detected[0]) if detected else np.nan,
        "recovery_steps": float(recovered[0]) if recovered else np.nan,
        "recovery_reference_upper": recovery_upper,
        "detected": bool(detected),
        "recovered": bool(recovered),
    }


def persistent_classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    *,
    phases: dict[str, tuple[int, ...]],
    labels: list[str],
    thresholds: np.ndarray | float,
    episode_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    predicted = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    if predicted.shape != expected.shape or expected.shape != known.shape:
        raise ValueError("Probabilities, targets, and masks must align")
    rows = []
    for phase in ("pre", "fault", "post"):
        indices = np.asarray(phases.get(phase, ()), dtype=np.int64)
        metrics: dict[str, Any] = {}
        if len(indices):
            metrics = phase2_multilabel_metrics(
                predicted[indices], expected[indices], known[indices], labels, thresholds
            )
        rows.append(
            {
                "run_id": run_id,
                "episode_id": episode_id,
                "phase": phase,
                "sample_count": int(len(indices)),
                "macro_f1": _nan_if_none(metrics.get("macro_f1")),
                "micro_f1": _nan_if_none(metrics.get("micro_f1")),
                "masked_bce": (
                    float(
                        np.nanmean(
                            masked_binary_cross_entropy_per_sample(
                                predicted[indices], expected[indices], known[indices]
                            )
                        )
                    )
                    if len(indices)
                    else np.nan
                ),
            }
        )
    return rows


def _macro_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray | float,
) -> float:
    value = phase2_multilabel_metrics(
        probabilities, targets, target_mask, labels, thresholds
    )["macro_f1"]
    return float(value) if value is not None else np.nan


def paired_user_bootstrap(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    user_ids: np.ndarray,
    *,
    labels: list[str],
    thresholds: np.ndarray | float,
    run_id_a: str,
    run_id_b: str,
    repeats: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    """Paired bootstrap over users; repeated users contribute repeated rows."""
    first = np.asarray(probabilities_a, dtype=np.float64)
    second = np.asarray(probabilities_b, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    users = np.asarray(user_ids).astype(str)
    shapes_align = (
        first.shape == second.shape == expected.shape == known.shape
    )
    if not shapes_align:
        raise ValueError("Both predictions, targets, and target masks must align")
    if len(users) != len(first) or repeats <= 0 or not run_id_a or not run_id_b:
        raise ValueError("User IDs, repeats, and run IDs must be valid")
    unique_users = np.asarray(sorted(set(users.tolist())))
    if not len(unique_users):
        raise ValueError("At least one user is required")
    indices_by_user = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    index_hasher = hashlib.sha256()
    for repeat in range(repeats):
        sampled_users = rng.choice(unique_users, size=len(unique_users), replace=True)
        sampled_indices = np.concatenate([indices_by_user[user] for user in sampled_users])
        index_hasher.update(",".join(map(str, sampled_indices.tolist())).encode("utf-8"))
        index_hasher.update(b"\n")
        metric_a = _macro_f1(
            first[sampled_indices],
            expected[sampled_indices],
            known[sampled_indices],
            labels,
            thresholds,
        )
        metric_b = _macro_f1(
            second[sampled_indices],
            expected[sampled_indices],
            known[sampled_indices],
            labels,
            thresholds,
        )
        draws.append(
            {
                "repeat": repeat,
                "macro_f1_a": metric_a,
                "macro_f1_b": metric_b,
                "delta_a_minus_b": metric_a - metric_b,
            }
        )
    deltas = np.asarray([row["delta_a_minus_b"] for row in draws], dtype=np.float64)
    valid_deltas = deltas[np.isfinite(deltas)]
    summary = {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "sampling_unit": "user",
        "metric": "macro_f1",
        "seed": int(seed),
        "repeats": int(repeats),
        "unique_user_count": int(len(unique_users)),
        "valid_repeat_count": int(len(valid_deltas)),
        "sample_index_sha256": index_hasher.hexdigest(),
        "mean_delta_a_minus_b": (
            float(valid_deltas.mean()) if len(valid_deltas) else np.nan
        ),
        "ci_2_5": (
            float(np.quantile(valid_deltas, 0.025)) if len(valid_deltas) else np.nan
        ),
        "ci_97_5": (
            float(np.quantile(valid_deltas, 0.975)) if len(valid_deltas) else np.nan
        ),
    }
    return {"summary": summary, "draws": draws}
