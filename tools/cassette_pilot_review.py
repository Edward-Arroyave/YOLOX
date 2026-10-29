"""Record a manually inspected pilot image; do not infer labels for other images."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT / 'revision_visual_piloto'
NAME = 'cassette_test_simple_single_2026-06_medical_order_20260617_165506_20261028_221444_0000.png'
CLASSES = {0: 'cassette', 1: 'negative', 2: 'positive', 7: 'ROI'}
# Coordinates measured on the original 488 x 960 image (x_min, y_min, x_max, y_max).
# Existing result classes are preserved; each ROI is linked to its exterior.
BOXES = [
    (0, (12, 0, 481, 959), 'cassette'),
    (2, (64, 320, 200, 594), 'left'),
    (2, (202, 320, 337, 594), 'middle'),
    (1, (301, 320, 438, 594), 'right'),
    (7, (92, 354, 142, 562), 'left'),
    (7, (230, 356, 278, 562), 'middle'),
    (7, (367, 360, 416, 565), 'right'),
]

def main():
    OUT.mkdir(exist_ok=False)
    for name in ('labels', 'annotations', 'previews'):
        (OUT / name).mkdir()
    with Image.open(ROOT / 'images' / NAME) as im:
        w, h = im.size
        if (w, h) != (488, 960):
            raise ValueError('Unexpected image dimensions')
        preview = im.convert('RGB')
    draw = ImageDraw.Draw(preview)
    annotations, lines, links = [], [], []
    for aid, (cid, (x0,y0,x1,y1), group) in enumerate(BOXES, 1):
        bw, bh = x1-x0, y1-y0
        lines.append(f'{cid} {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} {bw/w:.8f} {bh/h:.8f}')
        annotations.append({'id': aid, 'image_id': 1, 'category_id': cid,
                            'bbox': [x0,y0,bw,bh], 'area': bw*bh, 'iscrowd': 0})
        draw.rectangle((x0,y0,x1,y1), outline={0:'cyan',1:'lime',2:'red',7:'yellow'}[cid], width=2)
        draw.text((x0,y0), f'{cid}:{group}', fill={0:'cyan',1:'lime',2:'red',7:'yellow'}[cid])
        if cid == 7:
            links.append({'roi_annotation_id': aid, 'outer_annotation_id': {'left':2,'middle':3,'right':4}[group]})
    (OUT/'labels'/Path(NAME).with_suffix('.txt').name).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    coco = {'images':[{'id':1,'file_name':NAME,'width':w,'height':h}], 'annotations':annotations,
            'categories':[{'id':i,'name':n} for i,n in CLASSES.items()]}
    (OUT/'annotations'/'annotations.json').write_text(json.dumps(coco,indent=2),encoding='utf-8')
    (OUT/'roi_links.json').write_text(json.dumps(links,indent=2),encoding='utf-8')
    preview.save(OUT/'previews'/NAME)
    imgs = sorted((ROOT/'images').glob('*.png'))
    pending = [{'image':p.name,'status':'needs_review','reason':'Revisión visual pendiente'} for p in imgs[:20] if p.name != NAME]
    (OUT/'review.json').write_text(json.dumps({'pilot_size':20,'reviewed':1,'pending':pending},indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'changes.jsonl').write_text(json.dumps({'image':NAME,'status':'completed','changes':'Cajas exteriores ajustadas; tres ROI medidas visualmente; clases de resultado conservadas.'},ensure_ascii=False)+'\n',encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
