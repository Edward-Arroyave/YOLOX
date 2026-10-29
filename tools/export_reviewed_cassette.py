"""Export reviewed annotations to a clean YOLOX split without overwriting prior work."""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'yolox_revisado'
PILOT=ROOT/'revision_visual_piloto'
MANIFEST=ROOT/'revision_reanudable'/'review.json'
CLASSES={0:'cassette',1:'negative',2:'positive',7:'ROI'}
def main():
    if OUT.exists(): raise SystemExit(f'Output exists; refusing to overwrite: {OUT}')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    selected=[r for r in manifest['images'] if r['status'] in ('completed','completed_identical_copy')]
    coco=json.loads((PILOT/'annotations'/'annotations.json').read_text(encoding='utf-8'))
    imgs={im['file_name']:im for im in coco['images']}
    anns=defaultdict(list)
    for a in coco['annotations']: anns[a['image_id']].append(a)
    groups=defaultdict(list)
    for r in selected:
        if r['image'] not in imgs: raise ValueError('Missing COCO image: '+r['image'])
        groups[r['sha256']].append(r)
    group_list=list(groups.values()); random.Random(20261028).shuffle(group_list)
    target=round(len(selected)*0.70); dp={0:()}
    for i, group in enumerate(group_list):
        for n, ids in list(dp.items()):
            value=n+len(group)
            if value not in dp: dp[value]=ids+(i,)
    best=min(dp,key=lambda x:(abs(x-target),-x))
    train=set(dp[best]); counts={}
    for split in ('train','val'):
        dest=OUT/split
        for d in ('annotations','images','labels'): (dest/d).mkdir(parents=True,exist_ok=False)
        chosen=[r for i,g in enumerate(group_list) if (i in train)==(split=='train') for r in g]
        split_imgs=[]; split_anns=[]
        for r in chosen:
            name=r['image']; info=imgs[name]
            source=ROOT/'images'/name
            if not source.is_file(): raise ValueError('Missing image: '+name)
            shutil.copy2(source,dest/'images'/name)
            label=PILOT/'labels'/Path(name).with_suffix('.txt').name
            if not label.is_file(): raise ValueError('Missing label: '+name)
            converted=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                p=line.split()
                if len(p)!=5 or int(p[0]) not in CLASSES: raise ValueError('Invalid YOLO row: '+name)
                p[0]=str(3 if p[0]=='7' else int(p[0])); converted.append(' '.join(p))
            (dest/'labels'/label.name).write_text('\n'.join(converted)+'\n',encoding='utf-8')
            split_imgs.append(info); split_anns.extend(anns[info['id']])
        data={'images':split_imgs,'annotations':split_anns,'categories':[{'id':i,'name':name,'supercategory':'none'} for i,name in CLASSES.items()]}
        (dest/'annotations'/'annotations.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        yaml=(f'path: {OUT.as_posix()}\ntrain: train/images\nval: val/images\nnc: 4\n'
              'names:\n  0: cassette\n  1: negative\n  2: positive\n  3: ROI\n'
              '# IDs COCO: cassette=0, negative=1, positive=2, ROI=7. TXT usa ROI=3.\n')
        (dest/'data.yaml').write_text(yaml,encoding='utf-8')
        counts[split]=len(chosen)
    print(json.dumps({'output':str(OUT),'counts':counts,'groups':len(groups)},indent=2))

if __name__=='__main__': main()
