"""Rebalance reviewed dataset while keeping identical images in one split."""
import hashlib,json,shutil
from collections import defaultdict
from pathlib import Path

ROOT=Path(r'D:\Projects\IA\data_set\yolox_revisado')

def main():
    datasets={}; groups=defaultdict(list)
    for split in ('train','val'):
        base=ROOT/split; file=base/'annotations'/'annotations.json'
        data=json.loads(file.read_text(encoding='utf-8')); datasets[split]=data
        by_id=defaultdict(list)
        for ann in data['annotations']: by_id[ann['image_id']].append(ann)
        for info in data['images']:
            image=base/'images'/info['file_name']; label=base/'labels'/(Path(info['file_name']).stem+'.txt')
            if not image.is_file() or not label.is_file(): raise ValueError('Missing image or label')
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            groups[digest].append((split,info,by_id[info['id']]))
    for group in groups.values():
        if len({x[0] for x in group})!=1: raise ValueError('Duplicate images span partitions')
    group_list=list(groups.values()); states={0:(0,())}
    for i,group in enumerate(group_list):
        # Cost counts images that would move; retain least disruptive split for each size.
        old=group[0][0]; size=len(group); new={}
        for count,(cost,inds) in states.items():
            for new_count,new_cost,new_inds in ((count,cost+(size if old=='train' else 0),inds),
                                               (count+size,cost+(size if old=='val' else 0),inds+(i,))):
                if new_count not in new or new_cost<new[new_count][0]: new[new_count]=(new_cost,new_inds)
        states=new
    total=sum(len(g) for g in group_list); target=total*.7
    best=min(states,key=lambda n:(abs(n-target),states[n][0])); train=set(states[best][1])
    selected={'train':[],'val':[]}; moves=[]
    for i,group in enumerate(group_list):
        desired='train' if i in train else 'val'
        for old,info,annotations in group:
            selected[desired].append((info,annotations))
            if desired!=old:
                for folder,filename in (('images',info['file_name']),('labels',Path(info['file_name']).stem+'.txt')):
                    src=ROOT/old/folder/filename; dst=ROOT/desired/folder/filename
                    if dst.exists(): raise ValueError(f'Target exists: {dst}')
                    moves.append((src,dst))
    for src,dst in moves: shutil.move(str(src),str(dst))
    for split,rows in selected.items():
        data=datasets[split]; data['images']=[]; data['annotations']=[]
        for image_id,(info,annotations) in enumerate(rows,1):
            image=dict(info); image['id']=image_id; data['images'].append(image)
            for item in annotations:
                ann=dict(item); ann['id']=len(data['annotations'])+1; ann['image_id']=image_id
                data['annotations'].append(ann)
        (ROOT/split/'annotations'/'annotations.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({'train':len(selected['train']),'val':len(selected['val']),'target_train':target,'moved_files':len(moves)}))

if __name__=='__main__':main()
