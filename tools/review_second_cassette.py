"""Append manually reviewed image 0001 to the pilot without touching source data."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT/'revision_visual_piloto'
NAME = 'cassette_test_simple_single_2026-06_medical_order_20260617_165506_20261028_221444_0001.png'
# Individually measured against image 0001; original positive/negative classes retained.
BOXES = [(0,(4,8,479,949),'cassette'),(2,(63,326,196,582),'left'),
         (2,(198,326,331,582),'middle'),(1,(298,326,425,582),'right'),
         (7,(92,360,141,561),'left'),(7,(227,359,274,560),'middle'),
         (7,(360,357,407,558),'right')]

MEASUREMENTS = {
    '0001': BOXES,
    '0002': [(0,(7,2,487,949),'cassette'),(2,(64,325,198,581),'left'),
             (2,(199,325,334,582),'middle'),(1,(298,325,430,582),'right'),
             (7,(94,358,142,562),'left'),(7,(228,359,276,562),'middle'),
             (7,(361,359,409,563),'right')],
    '0003': [(0,(5,9,474,945),'cassette'),(2,(61,329,197,583),'left'),
             (2,(195,329,331,583),'middle'),(1,(297,329,424,583),'right'),
             (7,(92,361,140,562),'left'),(7,(226,360,274,562),'middle'),
             (7,(358,359,405,562),'right')],
}

# Each row is measured from the corresponding photograph, in pixels:
# cassette bounds; three strip bounds; top and bottom of the window housing.
VISUAL_MEASUREMENTS = {
 '0004': ((8,5,485,951), (95,360,143,562), (229,360,277,562), (363,359,411,562), (326,583)),
 '0005': ((14,6,484,945), (96,359,144,560), (229,360,278,560), (362,359,409,561), (326,582)),
 '0006': ((8,8,480,947), (95,360,143,561), (228,359,276,561), (361,359,409,560), (326,582)),
 '0007': ((3,0,487,959), (89,359,138,563), (228,359,277,562), (365,357,415,560), (324,586)),
 '0008': ((6,8,482,947), (94,361,141,561), (228,360,275,560), (360,359,407,561), (328,583)),
 '0009': ((8,10,481,946), (93,361,142,562), (227,361,275,562), (360,360,407,562), (328,583)),
 '0010': ((5,4,483,950), (92,360,140,562), (228,359,275,562), (361,358,409,561), (327,583)),
 '0011': ((5,2,486,952), (91,359,140,562), (227,359,275,562), (361,359,409,562), (325,583)),
 '0012': ((7,7,478,946), (94,360,142,563), (228,360,275,562), (357,360,404,562), (327,583)),
 '0013': ((5,7,481,947), (95,361,142,561), (228,361,275,562), (360,359,408,560), (328,584)),
 '0014': ((4,2,486,952), (91,356,139,562), (227,356,275,562), (363,357,411,562), (324,584)),
 '0015': ((3,2,481,954), (90,360,140,563), (227,360,275,563), (361,359,409,562), (327,584)),
 '0016': ((3,2,486,959), (89,358,138,563), (226,358,274,562), (362,358,410,562), (325,584)),
 '0017': ((9,12,477,945), (94,362,142,562), (227,360,275,562), (357,359,405,561), (328,583)),
 '0018': ((3,0,487,959), (92,359,140,562), (227,359,276,562), (363,358,411,562), (325,584)),
 '0019': ((8,4,480,947), (89,357,138,560), (227,357,276,560), (365,358,414,561), (323,584)),
 '0020': ((5,0,484,955), (92,359,140,562), (228,359,276,562), (364,359,412,561), (325,584)),
 '0021': ((9,7,481,945), (94,359,141,562), (228,359,275,562), (361,359,408,562), (327,583)),
 '0022': ((14,7,484,945), (96,358,143,560), (229,360,277,560), (363,359,410,561), (326,582)),
 '0023': ((5,0,486,956), (92,358,140,562), (228,358,276,562), (362,358,410,562), (324,584)),
 '0024': ((10,0,482,956), (94,359,142,562), (230,359,278,562), (366,359,414,562), (325,584)),
}

for n, (cassette, left, middle, right, (top,bottom)) in VISUAL_MEASUREMENTS.items():
    # Shared C/T marks lie between the windows; preserve their overlap.
    MEASUREMENTS[n] = [(0,cassette,'cassette'),
        (2,(left[0]-29,top,left[2]+56,bottom),'left'),
        (2,(middle[0]-29,top,middle[2]+56,bottom),'middle'),
        (1,(right[0]-62,top,right[2]+19,bottom),'right'),
        (7,left,'left'),(7,middle,'middle'),(7,right,'right')]

def main():
    global NAME, BOXES
    number = sys.argv[1] if len(sys.argv)>1 else '0001'
    BOXES = MEASUREMENTS[number]
    NAME = 'cassette_test_simple_single_2026-06_medical_order_20260617_165506_20261028_221444_' + number + '.png'
    coco_path = OUT/'annotations'/'annotations.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    if any(i['file_name']==NAME for i in coco['images']):
        raise SystemExit('Already reviewed; no files changed')
    with Image.open(ROOT/'images'/NAME) as original:
        if original.size != (488,960): raise ValueError('Unexpected dimensions')
        preview = original.convert('RGB')
    draw = ImageDraw.Draw(preview)
    iid = max(i['id'] for i in coco['images'])+1
    next_id = max(a['id'] for a in coco['annotations'])+1
    ids = {}; links = []; lines=[]
    for offset,(cid,(x0,y0,x1,y1),group) in enumerate(BOXES):
        aid=next_id+offset; bw=x1-x0; bh=y1-y0
        assert 0 <= x0 < x1 <= 488 and 0 <= y0 < y1 <= 960
        lines.append(f'{cid} {(x0+x1)/976:.8f} {(y0+y1)/1920:.8f} {bw/488:.8f} {bh/960:.8f}')
        coco['annotations'].append({'id':aid,'image_id':iid,'category_id':cid,'bbox':[x0,y0,bw,bh],'area':bw*bh,'iscrowd':0})
        if cid in (1,2): ids[group]=aid
        if cid==7: links.append({'roi_annotation_id':aid,'outer_annotation_id':ids[group]})
        color={0:'cyan',1:'lime',2:'red',7:'yellow'}[cid]
        draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
        draw.text((x0,y0),f'{cid}:{group}',fill=color)
    coco['images'].append({'id':iid,'file_name':NAME,'width':488,'height':960})
    label_path=OUT/'labels'/Path(NAME).with_suffix('.txt').name
    label_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    preview.save(OUT/'previews'/NAME)
    coco_path.write_text(json.dumps(coco,indent=2),encoding='utf-8')
    links_path=OUT/'roi_links.json'
    all_links=json.loads(links_path.read_text(encoding='utf-8')); all_links.extend(links)
    links_path.write_text(json.dumps(all_links,indent=2),encoding='utf-8')
    review_path=OUT/'review.json'; review=json.loads(review_path.read_text(encoding='utf-8'))
    was_pending = any(x['image']==NAME for x in review['pending'])
    review['pending']=[x for x in review['pending'] if x['image']!=NAME]
    if was_pending: review['reviewed']+=1
    review_path.write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT/'changes.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'image':NAME,'status':'completed','changes':'Cajas exteriores y tres ROI medidas visualmente; clases originales conservadas.'},ensure_ascii=False)+'\n')
    resume_path=ROOT/'revision_reanudable'/'review.json'
    resume=json.loads(resume_path.read_text(encoding='utf-8'))
    for record in resume['images']:
        if record['image']==NAME:
            record.update(status='completed',visual_validation=True,reviewed_label=str(label_path))
    resume_path.write_text(json.dumps(resume,indent=2,ensure_ascii=False),encoding='utf-8')
    print(str(OUT/'previews'/NAME))

if __name__=='__main__': main()
