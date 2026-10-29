"""Audit existing cassette labels without treating numeric checks as visual review."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT / 'revision_visual_piloto' / 'auditoria.json'
ALLOWED = {0, 1, 2, 7}

def read_yolo(path, width, height):
    rows, errors = [], []
    if not path.exists():
        return rows, ['missing_label_file']
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            p = line.split()
            if len(p) != 5:
                raise ValueError('expected_five_values')
            raw, x, y, w, h = map(float, p)
            cid = int(raw)
            if cid != raw or cid not in ALLOWED:
                raise ValueError('unknown_class')
            if not all(0 <= v <= 1 for v in (x,y,w,h)) or w <= 0 or h <= 0:
                raise ValueError('invalid_values')
            if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                errors.append(f'line_{number}:outside_image')
            rows.append((cid, (x-w/2)*width, (y-h/2)*height, w*width, h*height))
        except (ValueError, OverflowError) as exc:
            errors.append(f'line_{number}:{exc}')
    return rows, errors

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    coco = json.loads((ROOT/'annotations'/'annotations.json').read_text(encoding='utf-8-sig'))
    by_image = defaultdict(list)
    for a in coco['annotations']:
        by_image[a['image_id']].append(a)
    coco_images = {im['file_name']: im for im in coco['images']}
    records = []
    for path in sorted((ROOT/'images').glob('*.png')):
        with Image.open(path) as im:
            w,h = im.size
        rows, errors = read_yolo(ROOT/'labels'/f'{path.stem}.txt', w, h)
        image_record = coco_images.get(path.name)
        differences = []
        if image_record:
            existing = by_image[image_record['id']]
            for row in rows:
                matches = [a for a in existing if a['category_id'] == row[0] and max(abs(a['bbox'][i]-row[i+1]) for i in range(4)) < 1.0]
                if not matches:
                    differences.append({'source':'yolo_only','class_id':row[0],'bbox':list(row[1:])})
            for a in existing:
                matches = [row for row in rows if row[0] == a['category_id'] and max(abs(a['bbox'][i]-row[i+1]) for i in range(4)) < 1.0]
                if not matches:
                    differences.append({'source':'coco_only','class_id':a['category_id'],'bbox':a['bbox']})
        counts = Counter(row[0] for row in rows)
        records.append({'image':path.name, 'width':w, 'height':h, 'label_count':len(rows),
                        'class_counts':dict(counts), 'numeric_errors':errors,
                        'coco_present':bool(image_record), 'coco_differences':differences,
                        'review_status':'needs_review'})
    output = ROOT/'yolox_70_30'
    split_files = {split:{p.name:p for p in (output/split/'images').glob('*.png')} for split in ('train','val')}
    intersection = sorted(split_files['train'].keys() & split_files['val'].keys())
    hashes = defaultdict(set)
    for split, files in split_files.items():
        for path in files.values(): hashes[digest(path)].add(split)
    cross_hash = sum(len(splits)>1 for splits in hashes.values())
    coco_roi_tiny = 0
    for split in ('train','val'):
        data = json.loads((output/split/'annotations'/'annotations.json').read_text(encoding='utf-8-sig'))
        coco_roi_tiny += sum(a['category_id']==7 and a['bbox'][2] < 1 and a['bbox'][3] < 1 for a in data['annotations'])
    report = {'total_images':len(records), 'original_coco_images':len(coco['images']),
              'missing_label_files':sum('missing_label_file' in r['numeric_errors'] for r in records),
              'empty_label_files':sum(not r['label_count'] and 'missing_label_file' not in r['numeric_errors'] for r in records),
              'images_with_numeric_errors':sum(bool(r['numeric_errors']) for r in records),
              'images_with_coco_conflicts':sum(bool(r['coco_differences']) for r in records),
              'split':{'train':len(split_files['train']),'val':len(split_files['val']),
                       'same_name_both_splits':intersection,'identical_content_cross_split':cross_hash,
                       'invalid_subpixel_roi_in_coco':coco_roi_tiny}, 'images':records}
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'images'},indent=2))

if __name__ == '__main__': main()
