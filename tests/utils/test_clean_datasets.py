import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.clean_datasets import clean_contents, inventory, resolve_cleanup_target


class TestCleanDatasets(unittest.TestCase):
    def test_rejects_dangerous_targets(self):
        with patch("tools.clean_datasets.Path.home", return_value=Path.home()):
            with self.assertRaises(ValueError):
                resolve_cleanup_target(str(Path.home()))

    def test_inventory_and_clean_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "datasets"
            nested = target / "batch" / "images"
            nested.mkdir(parents=True)
            (nested / "image.png").write_bytes(b"1234")
            (target / "metadata.json").write_text("{}", encoding="utf-8")

            files, directories, total_bytes = inventory(target)
            self.assertEqual(files, 2)
            self.assertEqual(directories, 2)
            self.assertEqual(total_bytes, 6)

            clean_contents(target)
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
