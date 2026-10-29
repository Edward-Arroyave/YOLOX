"""Per-class COCO metrics, kept free of torch so they can be tested on their own.

Vision Label Studio spec 030 (RF-030-11): besides AP (IoU .50:.95) and AR, each class
gets AP50 (IoU .50), so the app can show the same per-class figure as Roboflow.
"""

import numpy as np

from yolox.model_card import number

IOU_050 = 0  # first IoU threshold of COCOeval (.50, .55, ..., .95)
AREA_ALL = 0
MAX_DETS_100 = -1


def _mean(values, enabled):
    valid = values[values > -1]
    return number(np.mean(valid)) if enabled and valid.size else None


def per_class_metrics(precision, recall, names, ap_enabled=True, ar_enabled=True):
    """`precision`/`recall` are COCOeval's `eval["precision"]` [T,R,K,A,M] and `eval["recall"]` [T,K,A,M]."""
    result = {}
    for idx, name in enumerate(names):
        result[name] = {
            "ap": _mean(precision[:, :, idx, AREA_ALL, MAX_DETS_100], ap_enabled),
            "ap50": _mean(precision[IOU_050, :, idx, AREA_ALL, MAX_DETS_100], ap_enabled),
            "ar": _mean(recall[:, idx, AREA_ALL, MAX_DETS_100], ar_enabled),
        }
    return result
