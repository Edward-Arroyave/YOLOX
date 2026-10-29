"""Transfer labels only when the new image is byte-identical to a reviewed image."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
PREV=ROOT/'yolox_revisado'
OUT=ROOT/'revision_2075'
NAMES=['cassette','negative','positive','dual_test_negative','dual_test_2_positive',
       'dual_test_1_positive','dual_test_both_positive','ROI']

def main():
    labels=OUT/'labels_verificadas'; previews=OUT/'previews_verificadas'
    if labels.exists() or previews.exists(): raise SystemExit('Already exists; no files changed')
    labels.mkdir(); previews.mkdir()
    sources={}
    for split in ('train','val'):
        for path in (PREV/split/'images').glob('*.png'):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            label=PREV/split/'labels'/f'{path.stem}.txt'
            if not label.exists(): raise ValueError(f'Missing {label}')
            previous=sources.get(digest)
            contents=label.read_text(encoding='utf-8')
            if previous and previous[1]!=contents:
                raise ValueError('Identical photographs have conflicting reviewed labels')
            sources[digest]=(path,contents)
    invpath=OUT/'inventory.json'; inv=json.loads(invpath.read_text(encoding='utf-8'))
    coco={'images':[],'annotations':[], 'categories':[{'id':i,'name':n,'supercategory':'none'} for i,n in enumerate(NAMES)]}
    for row in inv['images']:
        found=sources.get(row['sha256'])
        if not found: continue
        src,contents=found; target=ROOT/'images'/row['image']; w,h=row['width'],row['height']
        image_id=len(coco['images'])+1
        coco['images'].append({'id':image_id,'file_name':row['image'],'width':w,'height':h})
        label=labels/f'{target.stem}.txt'; label.write_text(contents,encoding='utf-8')
        with Image.open(target) as im: preview=im.convert('RGB')
        draw=ImageDraw.Draw(preview)
        for line in contents.splitlines():
            cid,x,y,bw,bh=map(float,line.split()); cid=int(cid)
            x0=(x-bw/2)*w; y0=(y-bh/2)*h; width=bw*w; height=bh*h
            coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':image_id,'category_id':cid,
                                        'bbox':[round(x0,3),round(y0,3),round(width,3),round(height,3)],
                                        'area':round(width*height,3),'iscrowd':0})
            color={0:'cyan',1:'lime',2:'red',7:'yellow'}[cid]
            draw.rectangle((x0,y0,x0+width,y0+height),outline=color,width=2)
        preview.save(previews/target.name)
        row.update(status='verified_identical_copy',reviewed_source=src.name,reviewed_label=str(label))
    (OUT/'annotations_verificadas.json').write_text(json.dumps(coco,indent=2),encoding='utf-8')
    inv['verified_identical_count']=len(coco['images'])
    invpath.write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"verified_identical_count={len(coco['images'])} annotations={len(coco['annotations'])}")

if __name__=='__main__': main()
