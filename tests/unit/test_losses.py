import unittest

import numpy as np

from robustsense.training.losses import masked_binary_cross_entropy


class MaskedLossTest(unittest.TestCase):
    def test_unknown_target_does_not_change_loss(self):
        logits = np.array([[0.2, -0.4]], dtype=float)
        targets_a = np.array([[1.0, np.nan]])
        targets_b = np.array([[1.0, 0.0]])
        mask = np.array([[True, False]])

        loss_a = masked_binary_cross_entropy(logits, targets_a, mask)
        loss_b = masked_binary_cross_entropy(logits, targets_b, mask)
        self.assertAlmostEqual(loss_a, loss_b, places=12)

    def test_all_unknown_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "every target is unknown"):
            masked_binary_cross_entropy(
                np.zeros((1, 1)), np.full((1, 1), np.nan), np.zeros((1, 1), dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
