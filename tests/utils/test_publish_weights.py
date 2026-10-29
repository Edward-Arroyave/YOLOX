import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import publish_weights

from tools.publish_weights import (
    class_names_from_metrics,
    embed_class_names,
    find_next_version,
    format_version,
    next_patch_version,
    normalize_prefix,
    sha256_file,
    version_number,
)


class TestPublishWeights(unittest.TestCase):
    def test_publication_contains_card_and_metrics_without_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_ckpt.pth"
            checkpoint.write_bytes(b"model")
            exp = root / "exp.py"
            exp.write_text("# experiment")
            (root / "model_report.html").write_text("# Model card")
            (root / "model_report.md").write_text("# Ficha")
            (root / "model_report.xlsx").write_bytes(b"excel report")
            (root / "metrics.json").write_text("{}")
            client = MagicMock()
            client.list_blobs.return_value = []
            argv = ["publish_weights", "--ckpt", str(checkpoint), "--exp-file", str(exp),
                    "--project", "lis_yolox", "--version", "1.0.0", "--report-dir", str(root)]
            with patch("sys.argv", argv), patch.object(publish_weights, "load_environment"), \
                    patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "test",
                                             "AZURE_STORAGE_CONTAINER": "models"}), \
                    patch.object(publish_weights, "create_container_client", return_value=client), \
                    patch.object(publish_weights, "export_onnx",
                                 side_effect=lambda ckpt, output, *args: output.write_bytes(b"onnx")):
                self.assertEqual(publish_weights.main(), 0)
            names = [call.kwargs["name"] for call in client.upload_blob.call_args_list]
            self.assertEqual(names, ["weights/1.0.0/best_ckpt.pth", "weights/1.0.0/lis_yolox.onnx",
                                     "weights/1.0.0/metrics.json", "weights/1.0.0/model_report.html",
                                     "weights/1.0.0/model_report.md",
                                     "weights/1.0.0/model_report.xlsx"])
            self.assertTrue(all(not call.kwargs["overwrite"] for call in client.upload_blob.call_args_list))

    def test_class_names_come_from_metrics_dataset_classes_by_id(self):
        """Vision Label Studio spec 035 r3 (CA-035-19): the model carries its class names."""
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.json"
            metrics.write_text('{"dataset": {"classes": {"10": "linea", "2": "test", "0": "cassette"}}}', encoding="utf-8")
            self.assertEqual(class_names_from_metrics(metrics), ["cassette", "test", "linea"])
            metrics.write_text("{}", encoding="utf-8")
            self.assertEqual(class_names_from_metrics(metrics), [])
            self.assertEqual(class_names_from_metrics(Path(directory) / "missing.json"), [])

    def test_embed_class_names_appends_onnx_metadata_props(self):
        """ModelProto field 14 (metadata_props): StringStringEntryProto {1: "names", 2: JSON list}."""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"\x08\x07")  # ir_version = 7
            embed_class_names(model, ["cassette", "test"])
            value = b'["cassette","test"]'
            entry = b"\x0a\x05names" + b"\x12" + bytes([len(value)]) + value
            self.assertEqual(model.read_bytes(), b"\x08\x07" + b"\x72" + bytes([len(entry)]) + entry)

    def test_embed_class_names_uses_varint_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"")
            names = ["clase_larga_%03d" % i for i in range(20)]
            embed_class_names(model, names)
            data = model.read_bytes()
            self.assertEqual(data[0], 0x72)
            self.assertTrue(data[1] & 0x80)  # the entry is longer than 127 bytes: two-byte varint
            self.assertIn(b"clase_larga_019", data)

    def test_published_onnx_carries_the_class_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_ckpt.pth"
            checkpoint.write_bytes(b"model")
            exp = root / "exp.py"
            exp.write_text("# experiment")
            for name in ("model_report.html", "model_report.md", "model_report.xlsx"):
                (root / name).write_bytes(b"x")
            (root / "metrics.json").write_text('{"dataset": {"classes": {"1": "test", "0": "cassette"}}}', encoding="utf-8")
            client = MagicMock()
            client.list_blobs.return_value = []
            uploaded = {}
            argv = ["publish_weights", "--ckpt", str(checkpoint), "--exp-file", str(exp),
                    "--project", "lis_yolox", "--version", "1.0.0", "--report-dir", str(root)]
            with patch("sys.argv", argv), patch.object(publish_weights, "load_environment"), \
                    patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "test",
                                             "AZURE_STORAGE_CONTAINER": "models"}), \
                    patch.object(publish_weights, "create_container_client", return_value=client), \
                    patch.object(publish_weights, "export_onnx",
                                 side_effect=lambda ckpt, output, *args: output.write_bytes(b"\x08\x07")), \
                    patch.object(publish_weights, "upload_file",
                                 side_effect=lambda c, source, blob: uploaded.__setitem__(blob, source.read_bytes())):
                self.assertEqual(publish_weights.main(), 0)
            self.assertIn(b'["cassette","test"]', uploaded["weights/1.0.0/lis_yolox.onnx"])
            self.assertTrue(uploaded["weights/1.0.0/lis_yolox.onnx"].startswith(b"\x08\x07"))

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
