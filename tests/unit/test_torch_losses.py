import torch

from robustsense.training.torch_losses import masked_weighted_bce_with_logits


def test_unknown_target_has_zero_gradient():
    logits = torch.tensor([[0.25, -0.75]], requires_grad=True)
    targets = torch.tensor([[1.0, float("nan")]])
    target_mask = torch.tensor([[True, False]])

    loss = masked_weighted_bce_with_logits(logits, targets, target_mask)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0
