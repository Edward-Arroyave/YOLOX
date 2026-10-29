"""VLS spec 039 r2 (CA-039-20): a base checkpoint must have the chosen model size.

YOLOX's load_ckpt skips layers whose shape differs, with only a warning: a YOLOX-s base under a YOLOX-m
experiment would train from (almost) scratch without anyone noticing.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pipeline import training_options
from tools.check_base_checkpoint import main, mismatched_layers

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def general_exp(directory: Path, chosen: dict, classes: int):
    annotations = directory / f"data-{classes}" / "training" / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "annotations.json").write_text(json.dumps({"categories": [{"id": i, "name": str(i)} for i in range(classes)]}))
    config = directory / f"config-{len(list(directory.iterdir()))}.json"
    config.write_text(json.dumps(training_options.effective(chosen)))
    environment = {"YOLOX_DATA_DIR": str(annotations.parents[1]), "YOLOX_TRAINING_CONFIG": str(config)}
    with patch.dict(os.environ, environment):
        spec = importlib.util.spec_from_file_location("general_for_check", REPOSITORY_ROOT / "exps/general/yolox_general.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Exp(), environment


class MismatchedLayersTest(unittest.TestCase):
    def test_compares_every_layer_but_the_class_predictions(self):
        model = {"backbone.a": torch.zeros(4, 3), "head.cls_preds.0.weight": torch.zeros(3, 8), "head.reg": torch.zeros(2)}
        same_size_other_classes = {"backbone.a": torch.zeros(4, 3), "head.cls_preds.0.weight": torch.zeros(5, 8),
                                   "head.reg": torch.zeros(2)}
        self.assertEqual(mismatched_layers(model, same_size_other_classes), [])
        other_size = {"backbone.a": torch.zeros(6, 3), "head.cls_preds.0.weight": torch.zeros(3, 8)}
        self.assertEqual(mismatched_layers(model, other_size), ["backbone.a", "head.reg"])


class CheckBaseCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        exp, _ = general_exp(self.root, {}, 3)
        self.checkpoint = self.root / "yolox_s.pth"
        torch.save({"model": exp.get_model().state_dict()}, self.checkpoint)

    def run_check(self, chosen, classes):
        _, environment = general_exp(self.root, chosen, classes)
        argv = ["check", "--exp-file", str(REPOSITORY_ROOT / "exps/general/yolox_general.py"), "--ckpt", str(self.checkpoint)]
        with patch.dict(os.environ, environment), patch("sys.argv", argv):
            return main()

    def test_same_size_with_other_classes_is_accepted(self):
        self.assertEqual(self.run_check({}, 5), 0)

    def test_other_model_size_is_refused(self):
        self.assertEqual(self.run_check({"modelSize": "m"}, 3), 1)


if __name__ == "__main__":
    unittest.main()
