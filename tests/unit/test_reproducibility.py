import unittest

from robustsense.utils.reproducibility import stable_seed


class ReproducibilityTest(unittest.TestCase):
    def test_stable_seed_is_repeatable_and_keyed(self):
        self.assertEqual(stable_seed(13, "sample", 4), stable_seed(13, "sample", 4))
        self.assertNotEqual(stable_seed(13, "sample", 4), stable_seed(13, "sample", 5))


if __name__ == "__main__":
    unittest.main()
