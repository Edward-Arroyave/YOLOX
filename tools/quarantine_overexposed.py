"""Quarantine photographs inspected visually and found unreadable."""
import json
import shutil
from pathlib import Path

OUT = Path(r'D:\Projects\IA\data_set\revision_reanudable')
PREFIX = 'cassette_test_simple_single_2026-07_medical_order_20260722_004643_20261028_221444_'
REJECTED = [PREFIX + suffix + '.png' for suffix in ('0019','0002','0020','0001','0009','0022')]
REJECTED += [
 'cassette_test_simple_single_2026-07_medical_order_20260715_094110_20261028_221444_0012.png',
 'cassette_test_simple_single_2026-07_medical_order_20260715_094110_20261028_221444_0021.png',
 'cassette_test_simple_single_2026-07_medical_order_20260722_004643_20261028_221444_0017.png',
]

def main():
    manifest_path = OUT/'review.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rejected = [r for r in manifest['images'] if r['image'] in REJECTED and r['status'] != 'excluded_overexposed']
    if sum(r['image'] in REJECTED for r in manifest['images']) != len(REJECTED):
        raise ValueError('Image list mismatch; no files moved')
    for row in rejected:
        src = OUT/row['split']/'images'/row['image']
        if not src.is_file():
            raise ValueError(f'Missing {src}; no files moved')
    folder = OUT/'excluidas_sobreexposicion'
    folder.mkdir(exist_ok=True)
    for row in rejected:
        src = OUT/row['split']/'images'/row['image']
        shutil.move(str(src), str(folder/row['image']))
        row['original_split'] = row.pop('split')
        row['status'] = 'excluded_overexposed'
        row['visual_validation'] = True
        row['reason'] = 'Sobreexposición intensa: límites y marcas de las ventanas se distinguen insuficientemente.'
    manifest['train'] = sum(r.get('split')=='train' for r in manifest['images'])
    manifest['val'] = sum(r.get('split')=='val' for r in manifest['images'])
    manifest['excluded'] = sum(r['status']=='excluded_overexposed' for r in manifest['images'])
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'exclusiones.json').write_text(json.dumps([r for r in manifest['images'] if r['status']=='excluded_overexposed'],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'excluded':len(rejected),'train':manifest['train'],'val':manifest['val']}))

if __name__ == '__main__': main()
