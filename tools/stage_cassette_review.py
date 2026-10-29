"""Create resumable cassette review workspace, without promoting proposals to labels."""
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT / 'revision_reanudable'
OLD = ROOT / 'yolox_70_30'
SEED = 20261028

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def select_groups(groups, target):
    # Subset sum keeps byte-identical photographs in a single partition.
    choices = {0: ()}
    for i, group in enumerate(groups):
        for count, indexes in list(choices.items()):
            total = count + len(group)
            if total <= target and total not in choices:
                choices[total] = indexes + (i,)
    best = max(choices)
    return set(choices[best])

def main():
    if OUT.exists():
        raise SystemExit(f'Output exists; refusing to overwrite: {OUT}')
    images = sorted((ROOT/'images').glob('*.png'))
    grouped = defaultdict(list)
    for image in images:
        grouped[digest(image)].append(image)
    groups = list(grouped.values())
    random.Random(SEED).shuffle(groups)
    validation_groups = select_groups(groups, round(len(images)*0.30))
    records = []
    for folder in ('train/images', 'val/images', 'proposals', 'previews'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups):
        split = 'val' if index in validation_groups else 'train'
        for path in group:
            shutil.copy2(path, OUT/split/'images'/path.name)
            original = ROOT/'labels'/f'{path.stem}.txt'
            source = original.read_text(encoding='utf-8-sig') if original.exists() else None
            legacy = next((p for p in (OLD/'train'/'labels'/f'{path.stem}.txt', OLD/'val'/'labels'/f'{path.stem}.txt') if p.exists()), None)
            candidate = legacy.read_text(encoding='utf-8-sig') if legacy else None
            item = {'image':path.name, 'split':split, 'sha256':digest(path),
                    'status':'needs_review', 'visual_validation':False,
                    'original_yolo':source, 'previous_unvalidated_yolo':candidate,
                    'reason':'Requiere comparación visual y resolución de clases y cajas.'}
            (OUT/'proposals'/f'{path.stem}.json').write_text(json.dumps(item, ensure_ascii=False, indent=2),encoding='utf-8')
            records.append({k:item[k] for k in ('image','split','sha256','status','visual_validation')})
    records.sort(key=lambda x:x['image'])
    pilot_name = 'cassette_test_simple_single_2026-06_medical_order_20260617_165506_20261028_221444_0000.png'
    # Transfer only the previously reviewed example; keep its source explicit.
    for r in records:
        if r['image'] == pilot_name:
            r['status'] = 'completed'
            r['visual_validation'] = True
            r['reviewed_label'] = str(ROOT/'revision_visual_piloto'/'labels'/Path(pilot_name).with_suffix('.txt').name)
    manifest = {'state':'incomplete_not_for_training', 'seed':SEED, 'total':len(records),
                'train':sum(r['split']=='train' for r in records), 'val':sum(r['split']=='val' for r in records),
                'classes':{'0':'cassette','1':'negative','2':'positive','7':'ROI'},
                'note':'Las propuestas no son etiquetas definitivas; imágenes idénticas permanecen juntas.',
                'images':records}
    (OUT/'review.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='images'},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
