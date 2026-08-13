import tempfile
import unittest
from pathlib import Path

from tools.publish_weights import (
    find_next_version,
    format_version,
    next_patch_version,
    normalize_prefix,
    sha256_file,
    version_number,
)


class TestPublishWeights(unittest.TestCase):
    def test_find_next_version(self):
        blobs = [
            "weights/1.0.0/best_ckpt.pth",
            "weights/1.0.0/vet_yolox.onnx",
            "weights/1.0.2/best_ckpt.pth",
            "other/99.0.0/file",
        ]
        self.assertEqual(find_next_version(blobs, "weights"), "1.0.3")
        self.assertEqual(find_next_version([], "weights"), "1.0.0")

    def test_version_format(self):
        self.assertEqual(format_version((1, 2, 7)), "1.2.7")
        self.assertEqual(version_number("1.2.7"), (1, 2, 7))
        self.assertEqual(next_patch_version("1.2.7"), "1.2.8")
        self.assertEqual(next_patch_version(None), "1.0.0")
        for invalid in ("7", "v1.0.0", "version_0001", "1.0"):
            with self.assertRaises(ValueError):
                version_number(invalid)

    def test_normalize_prefix(self):
        self.assertEqual(normalize_prefix("/weights/"), "weights")
        with self.assertRaises(ValueError):
            normalize_prefix("../weights")

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"model")
            self.assertEqual(
                sha256_file(path),
                "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
            )


if __name__ == "__main__":
    unittest.main()
