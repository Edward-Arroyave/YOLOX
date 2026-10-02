"""Live training progress for Vision Label Studio (progress-v1). No torch needed."""
import json
import tempfile
import unittest
from pathlib import Path

from yolox.training_progress import PROGRESS_ENV, progress_path, write_progress


class TestTrainingProgress(unittest.TestCase):
    def test_writes_progress_v1_with_only_the_contract_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "progress.json"
            history = [
                {"epoch": 1, "learning_rate": 0.01, "total_loss": 6.2, "iou_loss": 2.1,
                 "conf_loss": 3.0, "cls_loss": 1.1},  # l1_loss absent until the no-aug epochs
                {"epoch": 2, "learning_rate": 0.009, "total_loss": 5.0, "iou_loss": 1.9,
                 "conf_loss": 2.5, "cls_loss": 0.6, "l1_loss": 0.4},
            ]
            evaluations = [{"epoch": 2, "map_50_95": 0.41, "ap50": 0.7, "ap75": 0.43,
                            "average_recall": 0.5, "inference_ms": 3.2, "per_class": {"a": {}}}]
            written = write_progress(path, epoch=2, max_epoch=80, training_history=history,
                                     evaluations=evaluations, best=evaluations[0],
                                     started_at="2026-10-01T00:00:00+00:00")
            self.assertTrue(written)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "progress-v1")
            self.assertEqual(data["status"], "running")
            self.assertEqual((data["epoch"], data["maxEpoch"]), (2, 80))
            self.assertEqual(data["startedAt"], "2026-10-01T00:00:00+00:00")
            self.assertIsNotNone(data["updatedAt"])
            self.assertEqual(data["training_history"][0]["l1_loss"], None)
            self.assertEqual(data["training_history"][1]["l1_loss"], 0.4)
            self.assertEqual(set(data["training_history"][0]),
                             {"epoch", "learning_rate", "total_loss", "iou_loss", "conf_loss",
                              "cls_loss", "l1_loss"})
            self.assertEqual(data["evaluations"], [{"epoch": 2, "map_50_95": 0.41, "ap50": 0.7,
                                                    "ap75": 0.43, "average_recall": 0.5}])
            self.assertEqual(data["best"], data["evaluations"][0])
            self.assertEqual((data["version"], data["error"]), (None, None))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])  # atomic: no leftovers

    def test_non_finite_numbers_become_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            write_progress(path, epoch=1, max_epoch=2,
                           training_history=[{"epoch": 1, "total_loss": float("nan")}],
                           evaluations=[], best=None, started_at=None)
            data = json.loads(path.read_text(encoding="utf-8"))  # strict JSON (no NaN)
            self.assertIsNone(data["training_history"][0]["total_loss"])
            self.assertIsNone(data["best"])

    def test_never_breaks_training_when_it_cannot_write(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "file"
            blocker.write_text("x", encoding="utf-8")
            self.assertFalse(write_progress(blocker / "progress.json", epoch=1, max_epoch=1,
                                            training_history=[], evaluations=[], best=None,
                                            started_at=None))

    def test_the_path_comes_from_the_environment_and_is_optional(self):
        self.assertIsNone(progress_path({}))
        self.assertIsNone(progress_path({PROGRESS_ENV: "  "}))
        self.assertEqual(progress_path({PROGRESS_ENV: "/tmp/p.json"}), Path("/tmp/p.json"))


if __name__ == "__main__":
    unittest.main()
