import os
import io
import json
import subprocess
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
    def test_failed_test_still_publishes_report_and_keeps_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'lis_yolox' / 'best_ckpt.pth'
            environment = {
                'PIPELINE_BLOB_BASE_PREFIX': 'training', 'PIPELINE_WEIGHTS_PREFIX': 'weights',
                'AZURE_STORAGE_CONNECTION_STRING': 'mock', 'AZURE_STORAGE_CONTAINER': 'test',
                'PIPELINE_OUTPUT_DIR': directory,
            }
            client = MagicMock()
            client.list_blobs.return_value = []
            published = []

            def stage(name, command, dry_run, env):
                report_dir = Path(env['YOLOX_REPORT_DIR'])
                if name.startswith('4/6'):
                    report_dir.mkdir(parents=True)
                    checkpoint.write_bytes(b'best trained weights')
                    (report_dir / 'metrics.json').write_text(json.dumps({'best': {'map_50_95': 0.6862}}))
                    (report_dir / 'model_report.html').write_text('training report')
                    (report_dir / 'model_report.md').write_text('# Ficha')
                    (report_dir / 'model_report.xlsx').write_bytes(b'excel report')
                elif name.startswith('Evaluar modelo'):
                    raise subprocess.CalledProcessError(1, command)
                elif name.startswith('5/6'):
                    report = json.loads((report_dir / 'metrics.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['test_evaluation']['status'], 'failed')
                    self.assertEqual(report['best']['map_50_95'], 0.6862)
                    self.assertIn('non-zero exit status', (report_dir / 'model_report.html').read_text(encoding='utf-8'))
                    published.append(command)

            with patch.dict(os.environ, environment, clear=True), \
                 patch('sys.argv', ['pipeline', '--prefix', 'lis', '--skip-clean', '--allow-no-base']), \
                 patch('pipeline.run_training_pipeline.load_environment', return_value=None), \
                 patch('pipeline.run_training_pipeline.missing_runtime_dependencies', return_value=[]), \
                 patch('pipeline.run_training_pipeline.create_container_client', return_value=client), \
                 patch('pipeline.run_training_pipeline.run_stage', side_effect=stage), \
                 patch('pipeline.run_training_pipeline.clean_local_weights') as clean, \
                 patch('sys.stdout', new_callable=io.StringIO), \
                 patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(main(), 0)
            self.assertEqual(len(published), 1)
            clean.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b'best trained weights')

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

    def _dry_run(self, extra_argv):
        client = MagicMock()
        blob = MagicMock()
        blob.name = "weights/1.0.2/best_ckpt.pth"
        client.list_blobs.return_value = [blob]
        environment = {
            "PIPELINE_BLOB_BASE_PREFIX": "training", "PIPELINE_WEIGHTS_PREFIX": "weights",
            "AZURE_STORAGE_CONNECTION_STRING": "mock", "AZURE_STORAGE_CONTAINER": "yolox",
        }
        argv = ["pipeline", "--prefix", "lis", "--dataset-folder", "9-2026", "--dry-run", *extra_argv]
        with patch.dict(os.environ, environment, clear=True), patch("sys.argv", argv),              patch("pipeline.run_training_pipeline.load_environment", return_value=None),              patch("pipeline.run_training_pipeline.create_container_client", return_value=client),              patch("pipeline.run_training_pipeline.run_stage") as stage,              patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            code = main()
        return code, stage

    def _ingest_base_prefix(self, stage):
        ingest = next(c.args[1] for c in stage.call_args_list if c.args[0].startswith("3/6"))
        return ingest[ingest.index("--base-prefix") + 1]

    def test_dataset_prefix_replaces_the_env_base_prefix(self):
        # VLS spec 027 r8 (CA-027-30): the app sends the Blob prefix of the chosen dataset.
        code, stage = self._dry_run(["--dataset-prefix", "datasets/lis"])
        self.assertEqual(code, 0)
        self.assertEqual(self._ingest_base_prefix(stage), "datasets/lis")
        code, stage = self._dry_run(["--dataset-prefix", ""])
        self.assertEqual(code, 0)
        self.assertEqual(self._ingest_base_prefix(stage), "")

    def test_without_dataset_prefix_the_env_base_prefix_is_used(self):
        code, stage = self._dry_run([])
        self.assertEqual(code, 0)
        self.assertEqual(self._ingest_base_prefix(stage), "training")

    def test_unsafe_dataset_prefix_fails_before_any_stage(self):
        for prefix in ["../x", "/abs", "a\\b", "a/../b"]:
            with self.subTest(prefix=prefix):
                code, stage = self._dry_run(["--dataset-prefix", prefix])
                self.assertEqual(code, 1)
                stage.assert_not_called()

    def _stage_command(self, stage, prefix):
        return next(c.args[1] for c in stage.call_args_list if c.args[0].startswith(prefix))

    def test_training_options_reach_train_test_and_publish(self):
        # VLS spec 039 (CA-039-11).
        chosen = '{"epochs":80,"batchSize":4,"fp16":false,"learningRate":0.003,"ema":false,"mosaicScaleMin":0.8}'
        code, stage = self._dry_run(["--training-options", chosen])
        self.assertEqual(code, 0)
        train = self._stage_command(stage, "4/6")
        self.assertEqual(train[train.index("--batch-size") + 1], "4")
        self.assertNotIn("--fp16", train)
        self.assertEqual(train[-8:], ["max_epoch", "80", "basic_lr_per_img", "4.6875e-05", "ema", "false",
                                      "mosaic_scale", "0.8,1.1"])
        for prefix in ("Evaluar", "5/6"):
            command = self._stage_command(stage, prefix)
            self.assertEqual(command[command.index("--batch-size") + 1], "4")
            self.assertNotIn("--fp16", command)

    def test_without_training_options_the_commands_are_unchanged(self):
        code, stage = self._dry_run([])
        self.assertEqual(code, 0)
        train = self._stage_command(stage, "4/6")
        self.assertEqual(train[train.index("--batch-size") + 1], "8")
        self.assertIn("--fp16", train)
        self.assertEqual(train[-2], "output_dir")

    def test_invalid_training_options_fail_before_any_stage(self):
        # CA-039-12.
        for value in ['{"epochs":0}', '{"foo":1}', "no-json", "[]"]:
            with self.subTest(value=value):
                code, stage = self._dry_run(["--training-options", value])
                self.assertEqual(code, 1)
                stage.assert_not_called()

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
            side_effect=lambda module: None if module in {"thop", "onnxsim", "openpyxl"} else object(),
        ):
            self.assertEqual(missing_runtime_dependencies(False), ["openpyxl", "thop"])
            self.assertEqual(
                missing_runtime_dependencies(True), ["openpyxl", "thop", "onnx-simplifier"]
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
