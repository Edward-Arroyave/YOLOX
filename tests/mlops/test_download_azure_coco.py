import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import timezone
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "mlops" / "scripts" / "download_azure_coco.py"
SPEC = importlib.util.spec_from_file_location("download_azure_coco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestAzureCocoDownloadUtilities(unittest.TestCase):
    def test_parse_utc_and_half_open_boundaries(self):
        value = MODULE.parse_utc("2026-07-01T00:00:00Z", "start")
        self.assertEqual(value.tzinfo, timezone.utc)
        with self.assertRaises(ValueError):
            MODULE.parse_utc("2026-07-01T00:00:00", "start")

    def test_safe_local_path_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = MODULE.safe_local_path(root, "processed/train/image.png", "processed")
            self.assertEqual(safe, (root / "train" / "image.png").resolve())
            with self.assertRaises(ValueError):
                MODULE.safe_local_path(root, "processed/../secret.txt", "processed")

    def test_filter_coco_keeps_only_selected_images(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "a.png", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.png", "width": 10, "height": 10},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
                {"id": 11, "image_id": 2, "category_id": 1, "bbox": [0, 0, 2, 2]},
            ],
            "categories": [{"id": 1, "name": "cassette"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "train.json"
            annotation.write_text(json.dumps(coco), encoding="utf-8")
            counts = MODULE.filter_coco(annotation, {"a.png"}, annotation)
            filtered = json.loads(annotation.read_text(encoding="utf-8"))

        self.assertEqual(counts, {"images": 1, "annotations": 1})
        self.assertEqual([image["id"] for image in filtered["images"]], [1])
        self.assertEqual([ann["image_id"] for ann in filtered["annotations"]], [1])


if __name__ == "__main__":
    unittest.main()
