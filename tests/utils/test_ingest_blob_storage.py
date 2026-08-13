import tempfile
import unittest
from pathlib import Path

from tools.ingest_blob_storage import combine_prefix, normalize_blob_folder, safe_destination


class TestBlobStorageIngest(unittest.TestCase):
    def test_normalize_blob_folder(self):
        self.assertEqual(normalize_blob_folder("/projects\\batch-01/"), "projects/batch-01")
        for invalid in ("", "../secret", "folder/../../secret"):
            with self.assertRaises(ValueError):
                normalize_blob_folder(invalid)

    def test_combine_prefix(self):
        self.assertEqual(combine_prefix("incoming/", "/COCO"), "incoming/COCO")
        self.assertEqual(combine_prefix("", "COCO"), "COCO")

    def test_safe_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                safe_destination(root, "train/image.png"),
                (root / "train" / "image.png").resolve(),
            )
            with self.assertRaises(ValueError):
                safe_destination(root, "../image.png")


if __name__ == "__main__":
    unittest.main()
