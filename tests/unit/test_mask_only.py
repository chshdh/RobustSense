import inspect

import numpy as np
import torch

from robustsense.models.mask_only import (
    MaskOnlyModel,
    fit_mask_only_model,
    permute_availability_within_users,
)


def test_mask_only_api_can_receive_only_availability():
    parameters = list(inspect.signature(MaskOnlyModel.forward).parameters)
    assert parameters == ["self", "availability"]
    model = MaskOnlyModel(n_labels=2, hidden_dim=4)
    assert model(torch.ones((3, 6), dtype=torch.bool)).shape == (3, 2)


def test_within_user_permutation_is_reproducible_and_preserves_each_multiset():
    availability = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]
    )
    users = np.array(["a", "a", "b", "b"])
    first = permute_availability_within_users(availability, users, seed=13)
    second = permute_availability_within_users(availability, users, seed=13)
    assert np.array_equal(first, second)
    for user in ("a", "b"):
        selected = users == user
        assert sorted(map(tuple, first[selected])) == sorted(map(tuple, availability[selected]))


def test_mask_only_training_uses_masked_targets_and_is_reproducible():
    availability = np.tile(np.eye(6, dtype=np.float32), (2, 1))
    targets = (availability[:, :2].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    targets = np.concatenate([targets, 1 - targets], axis=1)
    target_mask = np.ones_like(targets, dtype=bool)
    target_mask[0, 1] = False
    first, first_history = fit_mask_only_model(
        availability,
        targets,
        target_mask,
        n_labels=2,
        epochs=5,
        seed=13,
    )
    second, second_history = fit_mask_only_model(
        availability,
        targets,
        target_mask,
        n_labels=2,
        epochs=5,
        seed=13,
    )
    assert first_history == second_history
    paired_parameters = zip(first.parameters(), second.parameters(), strict=True)
    for first_parameter, second_parameter in paired_parameters:
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
