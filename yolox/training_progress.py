"""Live training progress for Vision Label Studio (contract ``progress-v1``).

When ``VLS_PROGRESS_FILE`` is set, the trainer writes this JSON after every
epoch and every evaluation; the VM agent uploads it so the web app can draw
the charts while training. Field names are the same as ``metrics.json`` v1.
Without the variable nothing is written and training behaves as before.
This module does not import torch.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

PROGRESS_ENV = "VLS_PROGRESS_FILE"
HISTORY_FIELDS = ("epoch", "learning_rate", "total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss")
EVALUATION_FIELDS = ("epoch", "map_50_95", "ap50", "ap75", "average_recall")


def progress_path(env: Mapping[str, str] = os.environ) -> Path | None:
    value = (env.get(PROGRESS_ENV) or "").strip()
    return Path(value) if value else None


def _number(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    return value


def _pick(row: Mapping | None, fields: tuple[str, ...]) -> dict | None:
    if row is None:
        return None
    return {field: _number(row.get(field)) for field in fields}


def write_progress(path: Path | str, *, epoch: int, max_epoch: int, training_history, evaluations,
                   best, started_at: str | None) -> bool:
    """Writes the progress atomically. Returns False (never raises) if it cannot write."""
    path = Path(path)
    data = {
        "schema": "progress-v1",
        "status": "running",
        "epoch": epoch,
        "maxEpoch": max_epoch,
        "startedAt": started_at,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "training_history": [_pick(row, HISTORY_FIELDS) for row in training_history],
        "evaluations": [_pick(row, EVALUATION_FIELDS) for row in evaluations],
        "best": _pick(best, EVALUATION_FIELDS),
        "version": None,
        "error": None,
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return True
    except (OSError, ValueError):
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
