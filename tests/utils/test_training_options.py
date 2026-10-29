"""VLS spec 039: training options chosen in Vision Label Studio (closed catalog)."""
import importlib.util
import unittest
from pathlib import Path

from exps.cassette.settings import TRAIN_BATCH_SIZE
from pipeline import training_options as options
from yolox.exp.yolox_base import Exp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def cassette_exp():
    spec = importlib.util.spec_from_file_location("cassette_yolox", REPOSITORY_ROOT / "exps/cassette/cassette_yolox.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp()


class CatalogDefaultsTest(unittest.TestCase):
    """CA-039-13: the defaults are the current cassette experiment (the "estado actual")."""

    def test_defaults_match_the_cassette_experiment(self):
        exp = cassette_exp()
        defaults = options.defaults()
        self.assertEqual(defaults["epochs"], exp.max_epoch)
        self.assertEqual(defaults["batchSize"], TRAIN_BATCH_SIZE)
        self.assertAlmostEqual(defaults["learningRate"] / 64, exp.basic_lr_per_img)
        self.assertEqual(defaults["noAugEpochs"], exp.no_aug_epochs)
        self.assertEqual(defaults["warmupEpochs"], exp.warmup_epochs)
        self.assertEqual(defaults["minLrRatio"], exp.min_lr_ratio)
        self.assertEqual(defaults["weightDecay"], exp.weight_decay)
        self.assertEqual(defaults["momentum"], exp.momentum)
        self.assertEqual(defaults["ema"], exp.ema)
        self.assertEqual(defaults["evalInterval"], exp.eval_interval)
        self.assertEqual(defaults["mosaicProb"], exp.mosaic_prob)
        self.assertEqual((defaults["mosaicScaleMin"], defaults["mosaicScaleMax"]), exp.mosaic_scale)
        self.assertEqual(defaults["degrees"], exp.degrees)
        self.assertEqual(defaults["translate"], exp.translate)
        self.assertEqual(defaults["shear"], exp.shear)
        self.assertEqual(defaults["flipProb"], exp.flip_prob)
        self.assertEqual(defaults["mixupProb"], exp.mixup_prob)
        self.assertFalse(exp.enable_mixup)
        self.assertTrue(defaults["fp16"])
        self.assertEqual(len(defaults), 19)


class ValidateTest(unittest.TestCase):
    def test_accepts_catalog_values(self):
        chosen = {"epochs": 80, "learningRate": 0.003, "ema": False, "mosaicScaleMin": 0.8}
        self.assertEqual(options.validate(chosen), chosen)
        self.assertEqual(options.validate({}), {})
        self.assertEqual(options.validate({"degrees": 2}), {"degrees": 2.0})

    def test_rejects_unknown_keys_wrong_types_and_ranges(self):
        bad = [
            [], "epochs", None, {"foo": 1}, {"epochs": 2.5}, {"epochs": "50"}, {"epochs": True},
            {"epochs": 0}, {"epochs": 1001}, {"learningRate": float("inf")}, {"learningRate": float("nan")},
            {"ema": "false"}, {"ema": 0}, {"momentum": 1}, {"batchSize": 65},
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    options.validate(value)

    def test_rules_between_options_use_the_effective_values(self):
        for value in [{"epochs": 4}, {"epochs": 50}, {"warmupEpochs": 10, "epochs": 10},
                      {"mosaicScaleMin": 1.2}, {"mosaicScaleMin": 0.5, "mosaicScaleMax": 0.4}]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    options.validate(value)
        self.assertEqual(options.validate({"epochs": 60}), {"epochs": 60})


class TrainOverridesTest(unittest.TestCase):
    def test_exp_pairs_for_tools_train(self):
        pairs = options.exp_overrides({"epochs": 80, "batchSize": 4, "fp16": False, "learningRate": 0.003,
                                       "ema": False, "mosaicScaleMin": 0.8, "mixupProb": 0.3})
        self.assertEqual(pairs, ["max_epoch", "80", "basic_lr_per_img", "4.6875e-05", "ema", "false",
                                 "mosaic_scale", "0.8,1.1", "mixup_prob", "0.3", "enable_mixup", "true"])

    def test_describe(self):
        self.assertEqual(options.describe({}), "por defecto")
        self.assertEqual(options.describe({"epochs": 80, "ema": False}), "epochs=80, ema=false")


class MergeBooleanTest(unittest.TestCase):
    """CA-039-14: Exp.merge used to turn any text into True for a boolean."""

    def test_merge_parses_booleans(self):
        exp = Exp()
        exp.merge(["ema", "false"])
        self.assertIs(exp.ema, False)
        exp.merge(["ema", "true"])
        self.assertIs(exp.ema, True)
        exp.merge(["enable_mixup", "0"])
        self.assertIs(exp.enable_mixup, False)

    def test_merge_rejects_other_booleans(self):
        with self.assertRaises(ValueError):
            Exp().merge(["ema", "maybe"])


if __name__ == "__main__":
    unittest.main()
