"""Model-card collection and rendering; no inference or optional ML dependencies."""

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("map_50_95", "ap50", "ap75", "precision", "recall", "iou",
           "average_recall", "inference_ms", "fps", "validation_loss")


def number(value):
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def dataset_snapshot(exp):
    result = {"splits": {}, "classes": {}, "image_hashes": []}
    seen, hashes = set(), set()
    root = Path(getattr(exp, "data_dir", None) or "datasets/COCO")
    for split, attr in (("train", "train_ann"), ("val", "val_ann"), ("test", "test_ann")):
        ann = getattr(exp, attr, None)
        path = root / getattr(exp, "annotations_dir", "annotations") / (ann or "")
        if not ann or not path.is_file() or path.resolve() in seen:
            result["splits"][split] = None
            continue
        seen.add(path.resolve())
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result["splits"][split] = None
            continue
        classes = {str(c["id"]): c["name"] for c in data.get("categories", [])}
        result["classes"].update(classes)
        counts = Counter(str(a["category_id"]) for a in data.get("annotations", []))
        image_dir = getattr(exp, split + "_image_dir", split + "2017")
        split_hashes = []
        for im in data.get("images", []):
            image = root / image_dir / im.get("file_name", "{:012}.jpg".format(im["id"]))
            split_hashes.append(digest(image) if image.is_file() else None)
        hashes.update(h for h in split_hashes if h)
        result["splits"][split] = {
            "images": len(data.get("images", [])), "annotation_sha256": digest(path),
            "annotations": str(path), "image_hashes": split_hashes,
            "annotations_per_class": {name: counts[key] for key, name in classes.items()},
        }
    result["image_hashes"] = sorted(hashes)
    result["total_images"] = sum(s["images"] for s in result["splits"].values() if s) if any(result["splits"].values()) else None
    result["num_classes"] = len(result["classes"])
    result["identity_complete"] = all(
        all(s["image_hashes"]) for s in result["splits"].values() if s
    ) and bool(seen)
    return result


def comparison(report, base):
    old = (base or {}).get("best") or {}
    new = report.get("best") or {}
    rows = {}
    for key in METRICS:
        a, b = old.get(key), new.get(key)
        rows[key] = {"base": a, "new": b,
                     "difference": b - a if a is not None and b is not None else None}
    ds, previous = report["dataset"], (base or {}).get("dataset", {})
    growth = {"new_images": None, "existing_images": None, "growth_percent": None}
    if ds.get("identity_complete") and previous.get("identity_complete"):
        current, before = set(ds["image_hashes"]), set(previous["image_hashes"])
        growth = {"new_images": len(current - before), "existing_images": len(current & before),
                  "growth_percent": 100 * (len(current) - len(before)) / len(before) if before else None}
    return {"metrics": rows, "dataset": growth,
            "same_validation": bool(ds["splits"].get("val") and previous.get("splits", {}).get("val")
                and ds["splits"]["val"]["annotation_sha256"] == previous["splits"]["val"]["annotation_sha256"]
                and ds["splits"]["val"]["image_hashes"] == previous["splits"]["val"]["image_hashes"])}


def render(report):
    def show(value):
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            return "; ".join(f"{k}: {show(v)}" for k, v in value.items()) or "N/A"
        if isinstance(value, (list, tuple)):
            return ", ".join(show(v) for v in value) or "N/A"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Ficha técnica del modelo", "", "N/A: información no disponible.", ""]
    for section in ("model", "training", "hardware", "artifacts"):
        lines += ["## " + section, ""]
        lines += [f"- {key}: {show(value)}" for key, value in report.get(section, {}).items()]
        lines.append("")
    lines += ["## Dataset", "", "```json", json.dumps(report["dataset"], ensure_ascii=False, indent=2), "```", ""]
    # Keep the human document compact; content identities remain in metrics.json.
    start = lines.index("```json")
    compact = {k: v for k, v in report["dataset"].items() if k != "image_hashes"}
    compact["splits"] = {k: ({a: b for a, b in v.items() if a != "image_hashes"} if v else None)
                         for k, v in compact["splits"].items()}
    lines[start + 1] = json.dumps(compact, ensure_ascii=False, indent=2)
    lines += ["## Comparación del mejor checkpoint", "",
              "AP y AR en escala 0–1; diferencias absolutas. mAP = mAP 50:95.", "",
              "| Métrica | Modelo base | Modelo nuevo | Diferencia |", "|---|---|---|---|"]
    for key, row in report["comparison"]["metrics"].items():
        lines.append(f"| {key} | {show(row['base'])} | {show(row['new'])} | {show(row['difference'])} |")
    lines += ["", "## Pérdidas de entrenamiento", "", show(report.get("losses")),
              "", "Promedio por iteración de la última época, rank 0. No es validation loss ni IoU de detección.",
              "", "## Métricas por clase (mejor checkpoint)", "",
              show((report.get("best") or {}).get("per_class")), "", "## Conclusión", ""]
    delta = report["comparison"]["metrics"]["map_50_95"]["difference"]
    if delta is None:
        lines.append("No es posible determinar mejora: falta mAP del modelo base o nuevo.")
    else:
        status = "mejora" if delta > 0 else "empeora" if delta < 0 else "mantiene"
        lines.append(f"El mAP {status}; variación: {delta:+.6f} ({delta * 100:+.3f} puntos porcentuales).")
    if not report["comparison"]["same_validation"]:
        lines.append("No se ha verificado un conjunto de validación idéntico; la diferencia no demuestra una mejora comparable.")
    lines += [f"- {k}: {show(v)}" for k, v in report["comparison"]["dataset"].items()]
    per_class = (report.get("best") or {}).get("per_class", {})
    worst = sorted(((v["ap"], k) for k, v in per_class.items() if v.get("ap") is not None))[:3]
    lines.append("Clases con menor AP: " + (", ".join(f"{k} ({v:.4f})" for v, k in worst) or "N/A"))
    improvements = report["comparison"].get("class_improvements", [])
    lines.append("Clases con mayor mejora de AP: " + (", ".join(f"{k} ({v:+.4f})" for v, k in improvements) or "N/A"))
    lines += ["", "## Modelo base", "", show(report.get("base_model")), ""]
    return "\n".join(lines)


def write_report(report, directory, base=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("metrics.json", "model_card.md")):
        raise FileExistsError(f"Model report already exists in {directory}")
    report["schema_version"] = 1
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["base_model"] = {"model": (base or {}).get("model"),
                            "dataset_version": (base or {}).get("dataset", {}).get("version"),
                            "total_images": (base or {}).get("dataset", {}).get("total_images")}
    report["comparison"] = comparison(report, base)
    previous_classes = ((base or {}).get("best") or {}).get("per_class", {})
    improvements = []
    for name, values in (report.get("best") or {}).get("per_class", {}).items():
        old = previous_classes.get(name, {}).get("ap")
        new = values.get("ap")
        if old is not None and new is not None and new > old:
            improvements.append((new - old, name))
    report["comparison"]["class_improvements"] = sorted(improvements, reverse=True)[:3]
    for name, value in (("metrics.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)),
                        ("model_card.md", render(report))):
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(value)
