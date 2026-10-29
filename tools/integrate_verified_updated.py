"""Integrate byte-identical reviewed photographs into the existing YOLOX output."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'yolox_revisado'
REVIEW=ROOT/'revision_2075'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    inv=json.loads((REVIEW/'inventory.json').read_text(encoding='utf-8'))
    candidates=[x for x in inv['images'] if x['status']=='verified_identical_copy']
    split_data={}; hash_splits=defaultdict(set); names={}
    for split in ('train','val'):
        base=OUT/split
        coco_path=base/'annotations'/'annotations.json'
        coco=json.loads(coco_path.read_text(encoding='utf-8'))
        if {x['id'] for x in coco['categories']}!=set(range(8)): raise ValueError('Incorrect category schema')
        split_data[split]=(coco_path,coco)
        for item in coco['images']:
            name=item['file_name']; f=base/'images'/name
            if not f.is_file(): raise ValueError(f'Missing {f}')
            hash_splits[sha(f)].add(split)
            names[name]=split
    if any(len(s)>1 for s in hash_splits.values()): raise ValueError('Identical source images already cross splits')
    planned=[]
    for row in candidates:
        image=ROOT/'images'/row['image']; label=Path(row['reviewed_label'])
        if not image.is_file() or not label.is_file() or sha(image)!=row['sha256']:
            raise ValueError(f'Invalid source for {row["image"]}')
        if row['image'] in names: continue
        partitions=hash_splits.get(row['sha256'],set())
        if len(partitions)!=1: raise ValueError('Reviewed source has no unique partition')
        split=next(iter(partitions)); w,h=Image.open(image).size
        rows=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            p=line.split()
            if len(p)!=5: raise ValueError('Not YOLO detection format')
            cid=int(p[0]); x,y,bw,bh=map(float,p[1:])
            if cid not in (0,1,2,7) or min(x-bw/2,y-bh/2,bw,bh)<-1e-6 or x+bw/2>1.000001 or y+bh/2>1.000001:
                raise ValueError('Invalid class or geometry')
            rows.append((cid,x,y,bw,bh))
        planned.append((row,image,label,split,w,h,rows))
    for row,image,label,split,w,h,rows in planned:
        base=OUT/split; coco=split_data[split][1]
        image_id=max((x['id'] for x in coco['images']),default=0)+1
        next_id=max((x['id'] for x in coco['annotations']),default=0)+1
        shutil.copy2(image,base/'images'/image.name)
        shutil.copy2(label,base/'labels'/f'{image.stem}.txt')
        coco['images'].append({'id':image_id,'file_name':image.name,'width':w,'height':h})
        for cid,x,y,bw,bh in rows:
            x0=(x-bw/2)*w; y0=(y-bh/2)*h; boxw=bw*w; boxh=bh*h
            coco['annotations'].append({'id':next_id,'image_id':image_id,'category_id':cid,
                'bbox':[round(x0,3),round(y0,3),round(boxw,3),round(boxh,3)],
                'area':round(boxw*boxh,3),'iscrowd':0})
            next_id+=1
    for coco_path,coco in split_data.values(): coco_path.write_text(json.dumps(coco,indent=2),encoding='utf-8')
    print(json.dumps({'integrated':len(planned),'train':len(split_data['train'][1]['images']),
                      'val':len(split_data['val'][1]['images'])},indent=2))

if __name__=='__main__':main()
