import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.run_training_pipeline import (
    clean_local_weights,
    current_dataset_folder,
    experiment_path,
    env_bool,
    find_latest_version,
    normalize_model_prefix,
    required_env,
)


class TestTrainingPipeline(unittest.TestCase):
    def test_env_bool(self):
        for value in ("true", "1", "YES", "on"):
            with patch.dict(os.environ, {"FLAG": value}):
                self.assertTrue(env_bool("FLAG"))
        for value in ("false", "0", "NO", "off"):
            with patch.dict(os.environ, {"FLAG": value}):
                self.assertFalse(env_bool("FLAG", True))
        with patch.dict(os.environ, {"FLAG": "invalid"}):
            with self.assertRaises(ValueError):
                env_bool("FLAG")

    def test_required_env(self):
        with patch.dict(os.environ, {"REQUIRED_TEST": " value "}):
            self.assertEqual(required_env("REQUIRED_TEST"), "value")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                required_env("REQUIRED_TEST")

    def test_model_prefix(self):
        self.assertEqual(normalize_model_prefix("VET"), "vet")
        self.assertEqual(normalize_model_prefix("lis"), "lis")
        with self.assertRaises(ValueError):
            normalize_model_prefix("../vet")

    def test_experiment_path_template(self):
        path = experiment_path("exps/{prefix}/{project}.py", "vet", "vet_yolox")
        self.assertTrue(path.as_posix().endswith("exps/vet/vet_yolox.py"))
        with self.assertRaises(ValueError):
            experiment_path("exps/{unknown}.py", "vet", "vet_yolox")

    def test_current_dataset_folder(self):
        self.assertRegex(current_dataset_folder("America/Bogota"), r"^\d{1,2}-\d{4}$")
        with self.assertRaises(ValueError):
            current_dataset_folder("Invalid/Timezone")

    def test_find_latest_version(self):
        blobs = [
            "weights/1.0.0/best_ckpt.pth",
            "weights/1.0.0/vet_yolox.onnx",
            "weights/1.0.1/best_ckpt.pth",
            "weights/1.0.1/lis_yolox.onnx",
        ]
        self.assertEqual(find_latest_version(blobs, "weights"), "1.0.1")
        self.assertEqual(
            find_latest_version(blobs, "weights", "vet_yolox"), "1.0.0"
        )
        self.assertEqual(
            find_latest_version(blobs, "weights", "lis_yolox"), "1.0.1"
        )
        self.assertIsNone(find_latest_version(blobs, "weights", "other_yolox"))
        self.assertIsNone(find_latest_version([], "weights"))

    def test_clean_local_weights_preserves_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "vet_yolox"
            artifacts = root / "artifacts" / "vet_yolox"
            output.mkdir(parents=True)
            artifacts.mkdir(parents=True)
            (output / "best_ckpt.pth").write_bytes(b"model")
            (output / "train_log.txt").write_text("ok", encoding="utf-8")
            (artifacts / "base.pth").write_bytes(b"base")
            self.assertEqual(clean_local_weights(output, artifacts), 2)
            self.assertFalse((output / "best_ckpt.pth").exists())
            self.assertTrue((output / "train_log.txt").exists())
            self.assertFalse(artifacts.exists())


if __name__ == "__main__":
    unittest.main()
