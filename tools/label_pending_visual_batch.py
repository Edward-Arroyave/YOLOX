"""Apply individually measured visual labels to the requested dataset output."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'yolox_revisado'
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
PREFIX='cassette_test_simple_single_2026-07_medical_order_20260722_144702_20261029_000215_'
# Manually observed original image coordinates, not proportional ROI insets.
MEASURED={
 '0001':[(0,(0,0,458,913)),(1,(40,291,250,580)),(2,(194,289,400,580)),
          (7,(92,326,140,539)),(7,(296,323,342,537))],
 '0002':[(0,(4,0,455,913)),(1,(42,292,249,581)),(2,(194,288,399,581)),
          (7,(92,327,141,539)),(7,(295,323,341,537))],
 '0004':[(0,(0,0,462,921)),(1,(40,293,248,584)),(2,(195,290,398,582)),
          (7,(92,329,139,540)),(7,(296,324,341,538))],
}
EXCLUDED=('0000','0003')

def select_batch(name):
    global PREFIX, MEASURED, EXCLUDED
    if name=='first': return
    if name!='next_double': raise ValueError('Unknown batch')
    PREFIX='cassette_test_simple_single_2026-07_medical_order_20260722_150631_20261029_000215_'
    MEASURED={
      '0000':[(0,(12,14,457,908)),(1,(69,331,253,606)),(2,(206,333,404,606)),
               (7,(119,365,161,560)),(7,(306,368,349,562))],
      '0001':[(0,(3,14,460,914)),(1,(61,334,251,610)),(2,(204,333,406,610)),
               (7,(112,366,155,566)),(7,(304,366,348,566))],
      '0003':[(0,(7,9,459,917)),(1,(63,332,251,611)),(2,(204,329,403,611)),
               (7,(113,366,157,568)),(7,(304,362,348,565))],
      '0004':[(0,(5,10,461,915)),(1,(62,332,250,610)),(2,(204,332,405,610)),
               (7,(113,364,157,565)),(7,(304,365,348,565))],
    }
    EXCLUDED=('0002',)

def validate(rows,w,h):
    for c,(x0,y0,x1,y1) in rows:
        if c not in (0,1,2,7) or not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('Invalid class or geometry')

def prepare():
    for suffix,rows in MEASURED.items():
        name=PREFIX+suffix+'.png'
        with Image.open(ROOT/'images'/name) as original:
            w,h=original.size; validate(rows,w,h); preview=original.convert('RGB')
        draw=ImageDraw.Draw(preview)
        for c,box in rows:
            color={0:'cyan',1:'lime',2:'red',7:'yellow'}[c]
            draw.rectangle(box,outline=color,width=2)
            draw.text((box[0],box[1]),str(c),fill=color)
        dest=QA/('review_'+suffix+'.png'); preview.save(dest); print(dest)

def commit():
    path=OUT/'train'/'annotations'/'annotations.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    if {x['id'] for x in data['categories']}!=set(range(8)): raise ValueError('Expected eight categories')
    names={x['file_name'] for x in data['images']}
    invpath=ROOT/'revision_2075'/'inventory.json'; inv=json.loads(invpath.read_text(encoding='utf-8'))
    index={x['image']:x for x in inv['images']}
    prepared=[]
    for suffix,rows in MEASURED.items():
        name=PREFIX+suffix+'.png'; src=ROOT/'images'/name
        if name in names: raise ValueError('Already in output: '+name)
        with Image.open(src) as im: w,h=im.size
        validate(rows,w,h)
        if hashlib.sha256(src.read_bytes()).hexdigest()!=index[name]['sha256']: raise ValueError('Original image changed')
        if (OUT/'val'/'images'/name).exists(): raise ValueError('Image exists in val')
        prepared.append((name,src,rows,w,h))
    # Extend accepted cases only to current files with exactly the same SHA-256.
    all_prepared=[]
    for name,src,rows,w,h in prepared:
        all_prepared.append((name,src,rows,w,h,name))
        for other in inv['images']:
            if other['image']==name or other['image'] in names: continue
            if other['sha256']!=index[name]['sha256']: continue
            target=ROOT/'images'/other['image']
            if hashlib.sha256(target.read_bytes()).hexdigest()!=other['sha256']: raise ValueError('Duplicate changed')
            if (OUT/'val'/'images'/other['image']).exists(): raise ValueError('Duplicate already in val')
            all_prepared.append((other['image'],target,rows,w,h,name))
    for name,src,rows,w,h,reviewed_source in all_prepared:
        iid=max(x['id'] for x in data['images'])+1
        aid=max(x['id'] for x in data['annotations'])+1
        data['images'].append({'id':iid,'file_name':name,'width':w,'height':h})
        txt=[]; rowids=[]
        for c,(x0,y0,x1,y1) in rows:
            bw=x1-x0; bh=y1-y0; rowids.append(aid)
            data['annotations'].append({'id':aid,'image_id':iid,'category_id':c,
                'bbox':[x0,y0,bw,bh],'area':bw*bh,'iscrowd':0})
            txt.append(f'{c} {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} {bw/w:.8f} {bh/h:.8f}')
            aid+=1
        shutil.copy2(src,OUT/'train'/'images'/name)
        (OUT/'train'/'labels'/f'{src.stem}.txt').write_text('\n'.join(txt)+'\n',encoding='utf-8')
        index[name].update(status='completed_visual_review' if name==reviewed_source else 'completed_identical_copy',split='train',
            reviewed_label=str(OUT/'train'/'labels'/f'{src.stem}.txt'),
            visual_validation=name==reviewed_source,identical_to=reviewed_source,roi_links=[{'roi_annotation_id':rowids[3],'outer_annotation_id':rowids[1]},
                                             {'roi_annotation_id':rowids[4],'outer_annotation_id':rowids[2]}],
            result_basis='Marcas C/T compartidas: control visible; izquierda sin banda T visible, derecha con banda T visible; convención de etiquetado solicitada por el usuario.')
    for suffix in EXCLUDED:
        row=index[PREFIX+suffix+'.png']
        for other in inv['images']:
            if other['sha256']==row['sha256']:
                other.update(status='excluded_overexposed',visual_validation=other['image']==row['image'],
                    exclusion_reason='Pérdida de detalle del marco y marcas C/T por sobreexposición; excluida de la salida de entrenamiento.')
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    invpath.write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Etiquetadas e incorporadas: '+str(len(all_prepared))+'; destino: '+str(OUT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--commit',action='store_true'); parser.add_argument('--batch',default='first'); args=parser.parse_args()
    select_batch(args.batch)
    commit() if args.commit else prepare()
