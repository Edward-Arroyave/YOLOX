import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.evaluate_test import inspect_test, metric_rows
from yolox.model_report_html import render_test_section


class TestHoldoutEvaluation(unittest.TestCase):
    def test_dataset_detects_holdout_and_content_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = SimpleNamespace(data_dir=tmp, annotations_dir='.', num_classes=1)
            for name in ('train', 'val', 'test'):
                folder = root / name
                folder.mkdir()
                (folder / 'a.jpg').write_bytes(b'same' if name != 'val' else b'different')
                (folder / 'annotations.json').write_text(json.dumps({
                    'images': [{'id': 1, 'file_name': 'a.jpg'}],
                    'categories': [{'id': 1, 'name': 'cassette'}],
                    'annotations': [{'category_id': 1}]}))
                setattr(exp, name + '_ann', name + '/annotations.json')
                setattr(exp, name + '_image_dir', name)
            snapshot, overlap = inspect_test(exp)
            self.assertEqual(snapshot['total_images'], 3)
            self.assertEqual(overlap, {'train': 1, 'val': 0})
            (root / 'test/a.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'faltantes'):
                inspect_test(exp)
            (root / 'test/annotations.json').unlink()
            with self.assertRaisesRegex(ValueError, 'anotaciones'):
                inspect_test(exp)
            (root / 'test').rmdir()
            self.assertIsNone(inspect_test(exp)[1])

    def test_differences_zero_missing_and_html(self):
        new = {'map_50_95': 0.0, 'ap_small': None, 'per_class': {'<cassette>': {'ap': 0.0, 'ar': 0.0}}}
        base = {'map_50_95': 0.5, 'ap_small': None}
        rows = metric_rows(base, new)
        self.assertEqual(rows['map_50_95']['difference'], -0.5)
        self.assertIsNone(rows['ap_small']['difference'])
        self.assertIsNone(metric_rows(None, new)['map_50_95']['difference'])
        html = render_test_section({'test_evaluation': {'status': 'completed', 'base': base,
                            'new': new, 'metrics': rows, 'overlap_images': {'train': 1}}})
        self.assertIn('&lt;cassette&gt;', html)
        self.assertIn('optimistas', html)
        self.assertIn('-0.5', html)
