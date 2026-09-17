import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from exps.cassette.settings import TRAIN_BATCH_SIZE
from pipeline.run_training_pipeline import (
    REPOSITORY_ROOT,
    add_repository_to_pythonpath,
    clean_local_weights,
    current_dataset_folder,
    experiment_path,
    env_bool,
    find_latest_version,
    normalize_model_prefix,
    missing_runtime_dependencies,
    required_env,
    main,
)


class TestTrainingPipeline(unittest.TestCase):
    def test_base_selection_matches_training_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "yolox_s.pth"
            local.write_bytes(b"weights")
            for use_local in (True, False):
                with self.subTest(local=use_local):
                    client = MagicMock()
                    blob = MagicMock()
                    blob.name = "weights/1.0.2/best_ckpt.pth"
                    client.list_blobs.return_value = [blob]
                    argv = ["pipeline", "--prefix", "lis", "--dry-run"]
                    if use_local:
                        argv += ["--base-checkpoint", str(local)]
                    environment = {
                        "PIPELINE_BLOB_BASE_PREFIX": "training", "PIPELINE_WEIGHTS_PREFIX": "weights",
                        "AZURE_STORAGE_CONNECTION_STRING": "mock", "AZURE_STORAGE_CONTAINER": "test",
                    }
                    with patch.dict(os.environ, environment, clear=True), patch("sys.argv", argv), \
                         patch("pipeline.run_training_pipeline.load_environment", return_value=None), \
                         patch("pipeline.run_training_pipeline.create_container_client", return_value=client), \
                         patch("pipeline.run_training_pipeline.run_stage") as stage, \
                         patch("sys.stdout", new_callable=io.StringIO) as output:
                        self.assertEqual(main(), 0)
                    train = next(c.args for c in stage.call_args_list if c.args[0].startswith("4/6"))
                    publish = next(c.args for c in stage.call_args_list if c.args[0].startswith("5/6"))
                    test = next(c.args for c in stage.call_args_list if c.args[0].startswith("Evaluar modelo"))
                    self.assertEqual(test[1][test[1].index("--base-checkpoint") + 1],
                                     train[1][train[1].index("--ckpt") + 1])
                    stages = [c.args[0] for c in stage.call_args_list]
                    self.assertLess(stages.index(test[0]), stages.index(publish[0]))
                    command, env = train[1], train[3]
                    self.assertEqual(command.count("--ckpt"), 1)
                    self.assertNotIn("YOLOX_BASE_METRICS", env)
                    self.assertEqual(env["YOLOX_BASE_VERSION"], "none" if use_local else "1.0.2")
                    self.assertEqual(publish[1][publish[1].index("--base-version") + 1], env["YOLOX_BASE_VERSION"])
                    self.assertIn(env["YOLOX_BASE_SOURCE"], output.getvalue())
                    if use_local:
                        self.assertEqual(command[command.index("--ckpt") + 1], str(local.resolve()))
                        self.assertNotIn("best_ckpt.pth ->", output.getvalue())

    def test_hidden_checkpoint_override_is_rejected(self):
        for extra in ("--ckpt old.pth", "-cold.pth", "--ck=old.pth", "--resume"):
            with self.subTest(extra=extra), patch.dict(os.environ, {"PIPELINE_TRAIN_ARGS": extra}), \
                 patch("sys.argv", ["pipeline", "--prefix", "lis", "--dry-run"]), \
                 patch("pipeline.run_training_pipeline.load_environment", return_value=None), \
                 patch("sys.stderr", new_callable=io.StringIO) as output:
                self.assertEqual(main(), 1)
                self.assertIn("--base-checkpoint", output.getvalue())

    def test_shared_batch_size(self):
        self.assertEqual(TRAIN_BATCH_SIZE, 8)

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

    def test_shared_experiment_path(self):
        shared = experiment_path("exps/cassette/cassette_yolox.py")
        self.assertTrue(
            shared.as_posix().endswith("exps/cassette/cassette_yolox.py")
        )

    def test_current_dataset_folder(self):
        self.assertRegex(current_dataset_folder("America/Bogota"), r"^\d{1,2}-\d{4}$")
        with self.assertRaises(ValueError):
            current_dataset_folder("Invalid/Timezone")

    def test_add_repository_to_pythonpath(self):
        environment = {"PYTHONPATH": "existing/path"}
        add_repository_to_pythonpath(environment)
        entries = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[0], str(REPOSITORY_ROOT))
        self.assertIn("existing/path", entries)
        add_repository_to_pythonpath(environment)
        self.assertEqual(entries, environment["PYTHONPATH"].split(os.pathsep))

    def test_missing_runtime_dependencies(self):
        with patch(
            "pipeline.run_training_pipeline.importlib.util.find_spec",
            side_effect=lambda module: None if module in {"thop", "onnxsim"} else object(),
        ):
            self.assertEqual(missing_runtime_dependencies(False), ["thop"])
            self.assertEqual(
                missing_runtime_dependencies(True), ["thop", "onnx-simplifier"]
            )

    def test_find_latest_version(self):
        blobs = [
            "weights/1.0.0/best_ckpt.pth",
            "weights/1.0.0/vet_yolox.onnx",
            "weights/1.0.1/best_ckpt.pth",
            "weights/1.0.1/lis_yolox.onnx",
        ]
        self.assertEqual(find_latest_version(blobs, "weights"), "1.0.1")
        self.assertIsNone(find_latest_version([], "weights"))

    def test_latest_version_only_requires_checkpoint(self):
        blobs = [
            "weights/1.0.0/best_ckpt.pth",
            "weights/1.0.0/vet_yolox.onnx",
        ]
        self.assertEqual(find_latest_version(blobs, "weights"), "1.0.0")

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
