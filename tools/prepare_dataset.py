import json, random, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\Projects\IA\data_set")
OUT = SRC / "yolox_70_30"
SEED = 20261028
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_yaml_names(path):
    names = {}
    in_names = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "names:":
            in_names = True
            continue
        if line == "colors:":
            in_names = False
            continue
        if not in_names:
            continue
        if line.startswith('"') and ":" in line:
            k, v = line.split(":", 1)
            try: names[int(k.strip('" '))] = v.strip().strip('"')
            except ValueError: pass
    return names

def read_labels(path, w, h):
    rows = []
    if path.exists():
        for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            parts = line.split()
            if len(parts) != 5: raise ValueError(f"{path.name}:{ln}: expected 5 fields")
            vals = [float(x) for x in parts]
            cid = int(vals[0])
            if vals[0] != cid or cid < 0 or any(x < 0 or x > 1 for x in vals[1:]):
                raise ValueError(f"{path.name}:{ln}: invalid normalized YOLO values")
            x, y, bw, bh = vals[1:]
            x0 = max(0, (x - bw/2) * w); y0 = max(0, (y - bh/2) * h)
            rows.append({"class_id": cid, "yolo": vals, "bbox": [x0, y0, bw*w, bh*h]})
    return rows

def add_roi_rows(rows):
    """Derive ROI boxes from existing window/exterior boxes, never from pixels."""
    out = list(rows)
    for row in rows:
        if row["class_id"] == 0:
            continue
        cid, x, y, bw, bh = row["yolo"]
        roi = [7, x, y, bw * 0.40, bh * 0.78]
        out.append({"class_id": 7, "yolo": roi,
                    "bbox": [(x-roi[3]/2), (y-roi[4]/2), roi[3], roi[4]]})
    return out

def coco_dataset(items, categories, image_root, label_root):
    images, annotations = [], []
    for iid, (img_path, rows) in enumerate(items, 1):
        with Image.open(img_path) as im: w, h = im.size
        images.append({"id": iid, "file_name": img_path.name, "width": w, "height": h})
        for j, row in enumerate(rows, 1):
            x, y, bw, bh = row["bbox"]
            annotations.append({"id": len(annotations)+1, "image_id": iid,
                "category_id": row["class_id"], "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw*bh, 3), "iscrowd": 0})
    return {"info": {"description": "Validated YOLOX cassette dataset", "split": image_root},
            "licenses": [], "categories": categories, "images": images, "annotations": annotations}

def draw_preview(img_path, rows, dest, names):
    with Image.open(img_path).convert("RGB") as im:
        d = ImageDraw.Draw(im)
        w, h = im.size
        for row in rows:
            _, x, y, bw, bh = row["yolo"]
            x0, y0 = (x-bw/2)*w, (y-bh/2)*h
            x1, y1 = (x+bw/2)*w, (y+bh/2)*h
            color = (0, 225, 255) if row["class_id"] == 0 else ((255,0,0) if row["class_id"] == 2 else (0,200,0))
            d.rectangle((x0,y0,x1,y1), outline=color, width=max(2, min(w,h)//300))
            d.text((x0, max(0,y0-16)), names.get(row["class_id"], str(row["class_id"])), fill=color)
        im.save(dest)

def main():
    raise SystemExit("Generación deshabilitada: la salida anterior contiene ROI no revisadas y coordenadas COCO incorrectas. No sobrescribir el dataset.")
    names = {0: "cassette", 1: "negative", 2: "positive", 7: "ROI"}
    categories = [{"id": i, "name": names[i], "supercategory": "none"} for i in sorted(names)]
    records = []
    for img in sorted(p for p in (SRC/"images").iterdir() if p.suffix.lower() in EXTS):
        with Image.open(img) as im: w, h = im.size
        rows = add_roi_rows(read_labels(SRC/"labels"/(img.stem+".txt"), w, h))
        records.append((img, rows))
    # Deterministic stratification by annotation signature, preserving class balance.
    buckets = {}
    for rec in records:
        sig = tuple(sorted(r["class_id"] for r in rec[1]))
        buckets.setdefault(sig, []).append(rec)
    rng = random.Random(SEED)
    train, val = [], []
    for bucket in buckets.values():
        bucket = list(bucket); rng.shuffle(bucket)
        ntrain = round(len(bucket) * 0.70)
        if len(bucket) > 1: ntrain = min(max(1, ntrain), len(bucket)-1)
        train.extend(bucket[:ntrain]); val.extend(bucket[ntrain:])
    rng.shuffle(train); rng.shuffle(val)
    if OUT.exists(): shutil.rmtree(OUT)
    for split, items in (("train", train), ("val", val)):
        split_root = OUT / split
        (split_root/"images").mkdir(parents=True); (split_root/"labels").mkdir(parents=True); (split_root/"annotations").mkdir(parents=True)
        for img, rows in items:
            shutil.copy2(img, split_root/"images"/img.name)
            src_label = SRC/"labels"/(img.stem+".txt")
            label_lines = [" ".join(str(x) if i == 0 else f"{x:.6f}" for i, x in enumerate(row["yolo"])) for row in rows]
            (split_root/"labels"/(img.stem+".txt")).write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        (split_root/"annotations"/"annotations.json").write_text(json.dumps(coco_dataset(items, categories, split, str(split_root/"images")), indent=2), encoding="utf-8")
        (split_root/"data.yaml").write_text("path: .\ntrain: images\nval: images\nnc: 8\nnames:\n" + "\n".join(f"  {i}: {names[i]}" for i in sorted(names)) + "\n", encoding="utf-8")
    all_items = train + val
    (OUT/"annotations").mkdir(exist_ok=True)
    (OUT/"annotations"/"annotations.json").write_text(json.dumps(coco_dataset(all_items, categories, "all", str(OUT)), indent=2), encoding="utf-8")
    (OUT/"previews").mkdir()
    for img, rows in (train+val)[:20]: draw_preview(img, rows, OUT/"previews"/img.name, names)
    review_items = []
    for img, rows in records:
        nroi = sum(r["class_id"] == 7 for r in rows)
        review_items.append({"image": img.name, "status": "completed" if nroi else "needs_review",
            "roi_count": nroi, "reason": "ROI generated inside each existing window box; visual review required for edge/occlusion cases."})
    completed = sum(x["status"] == "completed" for x in review_items)
    review = {"pilot_size": min(20, len(records)), "status_counts": {"unchanged": 0, "corrected": 0, "completed": completed, "needs_review": len(records)-completed},
              "roi": {"status": "created", "class_id": 7, "note": "ROI boxes generated from existing window boxes using an inner-strip geometry rule; no pixels were changed and no masks were invented."},
              "items": review_items}
    (OUT/"review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT/"changes.jsonl").open("w", encoding="utf-8") as f:
        for img, _ in records: f.write(json.dumps({"image": img.name, "status": "unchanged", "changes": []}, ensure_ascii=False)+"\n")
    (OUT/"data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnc: 8\nnames:\n" + "\n".join(f"  {i}: {names[i]}" for i in sorted(names)) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "total": len(records), "train": len(train), "val": len(val), "labels": sum(bool(r) for _, r in records)}, indent=2))

if __name__ == "__main__": main()
