"""VLS spec 039: training options chosen in Vision Label Studio (closed catalog)."""
import json
import unittest
from pathlib import Path

from exps.cassette.settings import TRAIN_BATCH_SIZE
from pipeline import training_options as options
from yolox.exp.yolox_base import Exp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The cassette experiment before r2 (exps/cassette/cassette_yolox.py at 81db9f5): the "estado actual".
LEGACY_CASSETTE = {
    "epochs": 120, "batchSize": 8, "learningRate": 0.0015, "noAugEpochs": 60, "modelSize": "s", "inputSize": 640,
    "warmupEpochs": 5, "minLrRatio": 0.1, "weightDecay": 0.0005, "momentum": 0.9, "ema": True, "evalInterval": 5,
    "fp16": True, "mosaicProb": 0.25, "mosaicScaleMin": 0.9, "mosaicScaleMax": 1.1, "degrees": 1.0,
    "translate": 0.02, "shear": 0.0, "flipProb": 0.0, "mixupProb": 0.0,
}


class CatalogDefaultsTest(unittest.TestCase):
    """CA-039-13/18: the defaults are the «cassettes» profile, which is the former cassette experiment."""

    def test_defaults_come_from_the_cassettes_profile(self):
        profile = json.loads((REPOSITORY_ROOT / "exps/general/profiles/cassettes.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["name"], "Cassettes")
        self.assertEqual(options.defaults(), profile["options"])

    def test_cassettes_profile_is_the_former_cassette_experiment(self):
        self.assertEqual(options.defaults(), LEGACY_CASSETTE)
        self.assertEqual(options.defaults()["batchSize"], TRAIN_BATCH_SIZE)


class ValidateTest(unittest.TestCase):
    def test_accepts_catalog_values(self):
        chosen = {"epochs": 80, "learningRate": 0.003, "ema": False, "mosaicScaleMin": 0.8,
                  "modelSize": "m", "inputSize": 512}
        self.assertEqual(options.validate(chosen), chosen)
        self.assertEqual(options.validate({}), {})
        self.assertEqual(options.validate({"degrees": 2}), {"degrees": 2.0})

    def test_rejects_unknown_keys_wrong_types_and_ranges(self):
        bad = [
            [], "epochs", None, {"foo": 1}, {"epochs": 2.5}, {"epochs": "50"}, {"epochs": True},
            {"epochs": 0}, {"epochs": 1001}, {"learningRate": float("inf")}, {"learningRate": float("nan")},
            {"ema": "false"}, {"ema": 0}, {"momentum": 1}, {"batchSize": 65},
            {"modelSize": "xl"}, {"modelSize": 1}, {"inputSize": 500}, {"inputSize": 288}, {"inputSize": 1312},
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

    def test_effective_is_defaults_plus_options(self):
        effective = options.effective({"epochs": 80, "modelSize": "m"})
        self.assertEqual(effective["epochs"], 80)
        self.assertEqual(effective["modelSize"], "m")
        self.assertEqual(effective["batchSize"], 8)
        self.assertEqual(len(effective), 21)

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
