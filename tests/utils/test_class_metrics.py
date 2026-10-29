import unittest

import numpy as np

from yolox.class_metrics import per_class_metrics


def arrays():
    """COCOeval shapes: precision[T=10, R=101, K, A=4, M=3], recall[T=10, K, A=4, M=3]; -1 = no data."""
    precision = np.full((10, 101, 2, 4, 3), -1.0)
    recall = np.full((10, 2, 4, 3), -1.0)
    # class "a": 0.9 at IoU .50, 0.5 at every stricter IoU (only area=all, maxDets=100 counts).
    precision[0, :, 0, 0, -1] = 0.9
    precision[1:, :, 0, 0, -1] = 0.5
    recall[:, 0, 0, -1] = 0.6
    precision[0, :, 0, 1, -1] = 0.1  # another area range: must be ignored
    # class "b": half the recall points missing at IoU .50.
    precision[0, :50, 1, 0, -1] = 0.8
    precision[1:, :, 1, 0, -1] = 0.4
    return precision, recall


class PerClassMetricsTest(unittest.TestCase):
    def test_ca_030_09_ap50_is_iou_050_mean_ignoring_missing(self):
        precision, recall = arrays()
        result = per_class_metrics(precision, recall, ["a", "b"])
        self.assertAlmostEqual(result["a"]["ap50"], 0.9)
        self.assertAlmostEqual(result["b"]["ap50"], 0.8)

    def test_keeps_ap_over_all_ious_and_ar(self):
        precision, recall = arrays()
        result = per_class_metrics(precision, recall, ["a", "b"])
        self.assertAlmostEqual(result["a"]["ap"], (0.9 + 9 * 0.5) / 10)
        self.assertAlmostEqual(result["a"]["ar"], 0.6)
        self.assertIsNone(result["b"]["ar"])

    def test_disabled_metrics_are_null(self):
        precision, recall = arrays()
        result = per_class_metrics(precision, recall, ["a", "b"], ap_enabled=False, ar_enabled=False)
        self.assertEqual(result["a"], {"ap": None, "ap50": None, "ar": None})


if __name__ == "__main__":
    unittest.main()
