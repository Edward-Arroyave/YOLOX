"""VLS spec 039 r2: one general experiment, configured by YOLOX_TRAINING_CONFIG, classes from the dataset JSON."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import training_options

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_exp(relative):
    spec = importlib.util.spec_from_file_location(f"exp_{relative.replace('/', '_')}", REPOSITORY_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp()


class GeneralExpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def dataset(self, classes):
        annotations = self.root / "data" / "training" / "annotations"
        annotations.mkdir(parents=True)
        categories = [{"id": index, "name": f"clase-{index}"} for index in range(classes)]
        (annotations / "annotations.json").write_text(json.dumps({"images": [], "annotations": [], "categories": categories}))
        return str(self.root / "data")

    def config(self, chosen):
        path = self.root / "training-config.json"
        path.write_text(json.dumps(training_options.effective(chosen)))
        return str(path)

    def exp(self, environment, relative="exps/general/yolox_general.py"):
        with patch.dict(os.environ, environment):
            for name in ("YOLOX_TRAINING_CONFIG", "YOLOX_DATA_DIR"):
                if name not in environment:
                    os.environ.pop(name, None)
            return load_exp(relative)

    def test_classes_come_from_the_training_annotations(self):
        # CA-039-15.
        exp = self.exp({"YOLOX_DATA_DIR": self.dataset(5)})
        self.assertEqual(exp.num_classes, 5)

    def test_without_dataset_it_falls_back_to_three_classes(self):
        exp = self.exp({"YOLOX_DATA_DIR": str(self.root / "missing")})
        self.assertEqual(exp.num_classes, 3)

    def test_the_config_reaches_the_experiment(self):
        # CA-039-16.
        exp = self.exp({"YOLOX_DATA_DIR": self.dataset(3), "YOLOX_TRAINING_CONFIG": self.config(
            {"epochs": 80, "modelSize": "m", "inputSize": 512, "learningRate": 0.003, "batchSize": 4,
             "ema": False, "mosaicScaleMin": 0.8, "mixupProb": 0.3, "warmupEpochs": 3})})
        self.assertEqual(exp.max_epoch, 80)
        self.assertEqual((exp.depth, exp.width), (0.67, 0.75))
        self.assertEqual(exp.input_size, (512, 512))
        self.assertEqual(exp.test_size, (512, 512))
        self.assertEqual(exp.random_size, (10, 16))
        self.assertAlmostEqual(exp.basic_lr_per_img, 0.003 / 64)
        self.assertEqual(exp.batch_size, 4)
        self.assertIs(exp.ema, False)
        self.assertEqual(exp.mosaic_scale, (0.8, 1.1))
        self.assertEqual((exp.mixup_prob, exp.enable_mixup), (0.3, True))
        self.assertEqual(exp.warmup_epochs, 3)

    def test_nano_uses_depthwise_layers(self):
        exp = self.exp({"YOLOX_DATA_DIR": self.dataset(3), "YOLOX_TRAINING_CONFIG": self.config({"modelSize": "nano"})})
        self.assertEqual((exp.depth, exp.width), (0.33, 0.25))
        model = exp.get_model()
        self.assertTrue(any(type(module).__name__ == "DWConv" for module in model.modules()))

    def test_without_config_general_and_cassette_alias_match_the_former_cassette_experiment(self):
        # CA-039-17: the values of exps/cassette/cassette_yolox.py at 81db9f5.
        data_dir = self.dataset(3)
        for relative in ("exps/general/yolox_general.py", "exps/cassette/cassette_yolox.py"):
            with self.subTest(exp=relative):
                exp = self.exp({"YOLOX_DATA_DIR": data_dir}, relative)
                self.assertEqual(exp.num_classes, 3)
                self.assertEqual((exp.depth, exp.width, exp.act), (0.33, 0.50, "silu"))
                self.assertEqual((exp.input_size, exp.test_size, exp.random_size), ((640, 640), (640, 640), (14, 20)))
                self.assertEqual((exp.max_epoch, exp.batch_size, exp.warmup_epochs, exp.no_aug_epochs), (120, 8, 5, 60))
                self.assertEqual((exp.eval_interval, exp.print_interval, exp.save_history_ckpt), (5, 20, True))
                self.assertAlmostEqual(exp.basic_lr_per_img, 0.0015 / 64.0)
                self.assertEqual((exp.min_lr_ratio, exp.weight_decay, exp.momentum, exp.ema), (0.10, 5e-4, 0.9, True))
                self.assertEqual((exp.degrees, exp.translate, exp.shear, exp.perspective, exp.flip_prob),
                                 (1.0, 0.02, 0.0, 0.0, 0.0))
                self.assertEqual((exp.mosaic_prob, exp.mosaic_scale, exp.enable_mixup, exp.mixup_prob),
                                 (0.25, (0.9, 1.1), False, 0.0))
                self.assertEqual(exp.data_dir, data_dir)
                self.assertEqual((exp.train_ann, exp.val_ann, exp.test_ann), (
                    "training/annotations/annotations.json", "val/annotations/annotations.json",
                    "test/annotations/annotations.json"))
                self.assertEqual((exp.train_image_dir, exp.val_image_dir, exp.test_image_dir),
                                 ("training/images", "val/images", "test/images"))
                self.assertEqual(exp.data_num_workers, 2)
        self.assertEqual(self.exp({"YOLOX_DATA_DIR": data_dir}, "exps/cassette/cassette_yolox.py").exp_name, "cassette_yolox")

    def test_an_invalid_config_file_is_refused(self):
        path = self.root / "bad.json"
        path.write_text(json.dumps({"epochs": 0}))
        with self.assertRaises(ValueError):
            self.exp({"YOLOX_DATA_DIR": self.dataset(3), "YOLOX_TRAINING_CONFIG": str(path)})


if __name__ == "__main__":
    unittest.main()
