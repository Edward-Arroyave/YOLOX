"""Check training artifact integrity without asserting visual correctness."""
import hashlib,json,math
from collections import defaultdict,Counter
from pathlib import Path

ROOT=Path(r'D:\Projects\IA\data_set\yolox_revisado')
hash_splits=defaultdict(set);summary={};errors=[]
for split in ('train','val'):
    base=ROOT/split;data=json.loads((base/'annotations'/'annotations.json').read_text(encoding='utf-8'))
    if {c['id'] for c in data['categories']}!=set(range(8)):errors.append(split+': wrong classes')
    by_image=defaultdict(list)
    if len({a['id'] for a in data['annotations']})!=len(data['annotations']):errors.append(split+': repeated annotation ids')
    for a in data['annotations']:by_image[a['image_id']].append(a)
    names={};windows=Counter()
    for im in data['images']:
        names[im['file_name']]=im
        image=base/'images'/im['file_name'];label=base/'labels'/f'{Path(im["file_name"]).stem}.txt'
        if not image.is_file() or not label.is_file():errors.append(split+': missing pair '+im['file_name']);continue
        digest=hashlib.sha256(image.read_bytes()).hexdigest();hash_splits[digest].add(split)
        rows=[line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        annotations=by_image[im['id']];remaining=list(annotations)
        if len(rows)!=len(annotations):errors.append(split+': annotation count mismatch '+im['file_name'])
        for row in rows:
            if len(row)!=5:errors.append('Invalid label row');continue
            cid=int(row[0]);x,y,bw,bh=map(float,row[1:]);w,h=im['width'],im['height']
            if cid not in range(8) or not all(math.isfinite(v) for v in (x,y,bw,bh)):errors.append('Invalid class/number');continue
            box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
            match=next((a for a in remaining if a['category_id']==cid and max(abs(a['bbox'][i]-box[i]) for i in range(4))<.02),None)
            if match:remaining.remove(match)
            else:errors.append(split+': YOLO/COCO mismatch '+im['file_name'])
        windows[sum(a['category_id'] in (1,2) for a in annotations)]+=1
    actual={p.name for p in (base/'images').iterdir() if p.is_file()}
    if actual!=set(names):errors.append(split+': orphan/missing images')
    if len(list((base/'labels').glob('*.txt')))!=len(names):errors.append(split+': labels count mismatch')
    summary[split]={'images':len(names),'labels':len(names),'annotations':len(data['annotations']),
                    'window_counts':dict(windows),'categories':len(data['categories'])}
if any(len(s)>1 for s in hash_splits.values()):errors.append('Identical images across train/val')
inventory=json.loads((ROOT.parent/'revision_2075'/'inventory.json').read_text(encoding='utf-8'))
print(json.dumps({'output':summary,'current_2075_status':dict(Counter(r['status'] for r in inventory['images'])),
                  'integrity_errors':len(errors),'examples':errors[:5]},indent=2))
if errors:raise SystemExit(1)
