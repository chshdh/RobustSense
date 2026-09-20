import pytest

from robustsense.training.trainer import _verify_checkpoint_contract


def test_checkpoint_contract_mismatch_is_rejected():
    expected = {"version": 1, "model_name": "mlp", "schema": "abc"}
    actual = {"version": 1, "model_name": "mlp", "schema": "different"}

    with pytest.raises(ValueError, match="contract mismatch"):
        _verify_checkpoint_contract(expected, actual)
