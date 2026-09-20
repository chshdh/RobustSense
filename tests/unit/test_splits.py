import unittest

from robustsense.data.splits import assert_disjoint_split, user_level_split


class UserSplitTest(unittest.TestCase):
    def test_users_are_disjoint_and_complete(self):
        users = [f"u{i}" for i in range(10)]
        split = user_level_split(users, test_fold=0)
        combined = split["train"] + split["val"] + split["test"]
        self.assertEqual(set(combined), set(users))
        self.assertEqual(len(combined), len(set(combined)))

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            assert_disjoint_split({"train": ["u1"], "val": ["u2"], "test": ["u1"]})


if __name__ == "__main__":
    unittest.main()
