"""Save five individually observed cassette images for later validation."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'revision_2075'
PREFIX='cassette_test_simple_single_2026-07_medical_order_20260722_144529_20261029_000215_'
ROWS={
 '0000':None,
 '0001':((110,365,156,566),(303,365,349,566)),
 '0002':None,
 '0003':((109,364,155,566),(303,364,349,566)),
 '0004':((111,363,158,565),(303,364,350,565)),
}

def main():
    p=OUT/'inventory.json'; inventory=json.loads(p.read_text(encoding='utf-8'))
    items={r['image']:r for r in inventory['images']}
    proposals=OUT/'propuestas'; previews=OUT/'previews_propuestas'
    proposals.mkdir(exist_ok=True); previews.mkdir(exist_ok=True)
    for number,rois in ROWS.items():
        name=PREFIX+number+'.png'; row=items[name]
        if row['status']!='needs_visual_review': raise ValueError('Already processed '+name)
        if rois is None:
            row.update(status='excluded_overexposed',visual_review=True,
                       exclusion_reason='Marcas C/T y límites de ventana difícilmente distinguibles por exposición.')
            continue
        with Image.open(ROOT/'images'/name) as im:
            if im.size!=(464,960): raise ValueError('Unexpected dimensions')
            preview=im.convert('RGB')
        draw=ImageDraw.Draw(preview)
        windows=[(60,325,248,608),(207,325,405,608)]
        boxes=[]
        for i,(outer,roi) in enumerate(zip(windows,rois)):
            key='left' if i==0 else 'right'
            boxes.append({'id':key,'class_id':None,'candidate_class_id':1 if i==0 else 2,
                          'bbox_xyxy':outer,'reason':'Clase requiere revisión del protocolo FeLV/FIV.'})
            boxes.append({'id':f'roi_{key}','class_id':7,'bbox_xyxy':roi,'outer_id':key})
            draw.rectangle(outer,outline='orange',width=2); draw.rectangle(roi,outline='yellow',width=2)
        data={'image':name,'status':'needs_review','image_size':[464,960],
              'boxes':boxes,'observation':'Dos ventanas y marcas C/T compartidas; cajas exteriores y ROI propuestas.'}
        dest=proposals/(Path(name).stem+'.json'); dest.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
        preview.save(previews/name)
        row.update(status='needs_review_protocol_and_geometry',visual_review=True,proposal=str(dest))
    p.write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')
    print('reviewed=5 proposals=3 excluded=2')

if __name__=='__main__':main()
