"""Leakage-safe user-level fold allocation."""

from __future__ import annotations

from collections.abc import Iterable


def user_level_split(
    users: Iterable[str], test_fold: int, n_folds: int = 5
) -> dict[str, list[str]]:
    unique_users = sorted(set(users))
    if n_folds < 3:
        raise ValueError("At least three folds are required for train/val/test")
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be in [0, {n_folds - 1}]")

    val_fold = (test_fold + 1) % n_folds
    buckets = {index: [] for index in range(n_folds)}
    for index, user in enumerate(unique_users):
        buckets[index % n_folds].append(user)

    split = {
        "train": [
            u
            for fold, values in buckets.items()
            if fold not in {test_fold, val_fold}
            for u in values
        ],
        "val": buckets[val_fold],
        "test": buckets[test_fold],
    }
    assert_disjoint_split(split)
    return split


def assert_disjoint_split(split: dict[str, list[str]]) -> None:
    train, val, test = (set(split[name]) for name in ("train", "val", "test"))
    if train & val or train & test or val & test:
        raise ValueError("Subject leakage detected: train/val/test users overlap")
    if not train or not val or not test:
        raise ValueError("Each split must contain at least one user")
