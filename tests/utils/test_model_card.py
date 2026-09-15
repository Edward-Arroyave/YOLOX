import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yolox.model_card import comparison, dataset_snapshot, write_report


class TestModelCard(unittest.TestCase):
    def test_dataset_identity_and_test_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images/a.jpg").write_bytes(b"image content")
            data = {"images": [{"id": 1, "file_name": "a.jpg"}],
                    "categories": [{"id": 3, "name": "control"}],
                    "annotations": [{"category_id": 3}]}
            (root / "train.json").write_text(json.dumps(data))
            exp = SimpleNamespace(data_dir=tmp, annotations_dir=".", train_ann="train.json",
                                  val_ann="train.json", test_ann="train.json", train_image_dir="images")
            snapshot = dataset_snapshot(exp)
            self.assertEqual(snapshot["total_images"], 1)
            self.assertIsNone(snapshot["splits"]["test"])
            self.assertEqual(snapshot["splits"]["train"]["annotations_per_class"], {"control": 1})
            self.assertTrue(snapshot["identity_complete"])

    def test_comparison_missing_and_growth(self):
        ds = {"splits": {}, "identity_complete": True, "image_hashes": ["a", "b"]}
        report = {"dataset": ds, "best": {"map_50_95": .6}}
        base = {"dataset": {**ds, "image_hashes": ["a"]}, "best": {"map_50_95": .5}}
        result = comparison(report, base)
        self.assertAlmostEqual(result["metrics"]["map_50_95"]["difference"], .1)
        self.assertIsNone(result["metrics"]["precision"]["difference"])
        self.assertEqual(result["dataset"], {"new_images": 1, "existing_images": 1, "growth_percent": 100})
        self.assertIsNone(comparison(report, None)["dataset"]["new_images"])

    def test_persistent_report_no_overwrite_and_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {"model": {"version": "1.0.1"}, "dataset": {"splits": {}},
                      "best": {"map_50_95": .8, "epoch": 2},
                      "evaluations": [{"map_50_95": .8}, {"map_50_95": .3}]}
            write_report(report, tmp)
            saved = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertEqual(saved["comparison"]["metrics"]["map_50_95"]["new"], .8)
            self.assertIn("N/A", (Path(tmp) / "model_card.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_report(report, tmp)

    def test_failed_training_does_not_generate_card(self):
        # Execute the actual orchestration method without requiring CUDA/PyTorch.
        tree = ast.parse(Path("yolox/core/trainer.py").read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        namespace = {"logger": SimpleNamespace(error=lambda *a: None)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer.py", "exec"), namespace)
        calls = []
        fake = SimpleNamespace(rank=0, before_train=lambda: None,
                               train_in_epoch=lambda: None,
                               write_model_card=lambda: calls.append("card"),
                               after_train=lambda: calls.append("cleanup"))
        namespace["train"](fake)
        self.assertEqual(calls, ["card", "cleanup"])
        calls.clear()
        def fail():
            raise RuntimeError("training failed")
        fake.train_in_epoch = fail
        with self.assertRaises(RuntimeError):
            namespace["train"](fake)
        self.assertEqual(calls, ["cleanup"])


if __name__ == "__main__":
    unittest.main()
