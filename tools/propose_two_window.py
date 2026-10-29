"""Generate explicitly provisional ROI proposals for visually inspected images."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT/'revision_reanudable'
STEMS = {
 '20260715_094110': ('cassette_test_simple_single_2026-07_medical_order_20260715_094110_20261028_221444_0000.png',[(98,355,145,568),(297,356,343,570)]),
 '20260722_004643': ('cassette_test_simple_single_2026-07_medical_order_20260722_004643_20261028_221444_0005.png',[(110,337,157,547),(306,334,352,544)]),
 '20260722_005009': ('cassette_test_simple_single_2026-07_medical_order_20260722_005009_20261028_221444_0000.png',[(112,363,159,572),(309,360,355,569)]),
}
BAD = {'20260715_094110_0012','20260715_094110_0021','20260722_004643_0017'}

def main():
    path=OUT/'review.json'; review=json.loads(path.read_text(encoding='utf-8'))
    templates={}
    for key,(name,boxes) in STEMS.items():
        source=cv2.imread(str(ROOT/'images'/name),cv2.IMREAD_GRAYSCALE)
        templates[key]=[(source[y0:y1,x0:x1],(x0,y0,x1,y1)) for x0,y0,x1,y1 in boxes]
    count=0
    for row in review['images']:
        if row['status']!='needs_review': continue
        key=next((k for k in STEMS if k in row['image']),None)
        if key is None: continue
        suffix=row['image'][-8:-4]
        if key+'_'+suffix in BAD: continue
        name=row['image']; gray=cv2.imread(str(ROOT/'images'/name),cv2.IMREAD_GRAYSCALE)
        with Image.open(ROOT/'images'/name) as im: preview=im.convert('RGB'); w,h=im.size
        proposals=[]; draw=ImageDraw.Draw(preview)
        for index,(template,(x0,y0,x1,y1)) in enumerate(templates[key]):
            l=max(0,x0-24); t=max(0,y0-24); r=min(w,x1+24); b=min(h,y1+24)
            scores=cv2.matchTemplate(gray[t:b,l:r],template,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(scores)
            xx=l+loc[0]; yy=t+loc[1]; box=[xx,yy,xx+x1-x0,yy+y1-y0]
            proposals.append({'id':f'roi_{index+1}','class_id':7,'bbox_xyxy':box,'score_template':round(score,4),
                              'outer_id':f'window_{index+1}'})
            draw.rectangle(box,outline='yellow',width=2)
        # Shared center marks are included on both outer boxes.
        a,b=proposals[0]['bbox_xyxy'],proposals[1]['bbox_xyxy']
        middle=(a[2]+b[0])//2
        out1=[max(0,a[0]-50),max(0,a[1]-40),min(w,middle+22),min(h,a[3]+45)]
        out2=[max(0,middle-22),max(0,b[1]-40),min(w,b[2]+55),min(h,b[3]+45)]
        for index,box in enumerate((out1,out2)):
            proposals.append({'id':f'window_{index+1}','class_id':None,'bbox_xyxy':box,
                              'candidate_class_id':1 if index==0 else 2,
                              'reason':'Resultado candidato; falta protocolo de interpretación del casete FeLV/FIV.'})
            draw.rectangle(box,outline='orange',width=2)
        entry={'image':name,'image_size':[w,h], 'status':'needs_review',
               'visual_observation':'Dos ventanas visibles y marcas C/T compartidas; una banda izquierda, dos bandas derecha.',
               'reason':'Propuesta geométrica automática: confirmar límites; resultado sin protocolo.', 'boxes':proposals}
        (OUT/'proposals'/f'{Path(name).stem}.visual.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
        preview.save(OUT/'previews'/name)
        row.update(status='needs_review_protocol_and_geometry',visual_validation=False,
                   proposal=str(OUT/'proposals'/f'{Path(name).stem}.visual.json'))
        count+=1
    path.write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'propuestas={count}')

if __name__=='__main__': main()
