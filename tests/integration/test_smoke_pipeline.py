import tempfile
import unittest
from pathlib import Path

from robustsense.pipeline import evaluate, prepare, train


class SmokePipelineTest(unittest.TestCase):
    def test_prepare_train_evaluate(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative in (
                "configs/data/synthetic.yaml",
                "configs/experiment/dev.yaml",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    (source_root / relative).read_text(encoding="utf-8"), encoding="utf-8"
                )

            prepared = prepare(root / "configs/data/synthetic.yaml", fold=0)
            self.assertTrue(Path(prepared["dataset_path"]).is_file())
            run_dir = train(root, model_name="early", fold=0, seed=13, profile="dev")
            metrics = evaluate(run_dir, suite="smoke")

            self.assertTrue((run_dir / "best_checkpoint.npz").is_file())
            self.assertTrue((run_dir / "test_metrics.json").is_file())
            self.assertGreater(metrics["known_target_count"], 0)
            self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
