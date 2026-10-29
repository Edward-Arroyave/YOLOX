"""Validate and integrate visual decisions into the single requested output."""
import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict,Counter
from pathlib import Path
from PIL import Image

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'yolox_revisado'
INV=ROOT/'revision_2075'/'inventory.json'
NAMES=['cassette','negative','positive','dual_test_negative','dual_test_2_positive',
       'dual_test_1_positive','dual_test_both_positive','ROI']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def session(name):return name.rsplit('_',1)[0]

def validate(entry,source):
    if not entry.get('visually_reviewed'):raise ValueError('Missing visual review: '+entry['image'])
    if sha(source)!=entry['sha256']:raise ValueError('Hash mismatch: '+entry['image'])
    with Image.open(source) as im: dimensions=im.size
    if dimensions!=(entry['width'],entry['height']):raise ValueError('Dimension mismatch')
    if entry['decision']!='accepted':return
    boxes=entry['boxes']; identifiers=[b['id'] for b in boxes]
    if len(set(identifiers))!=len(identifiers):raise ValueError('Duplicate box identifiers')
    by_id={b['id']:b for b in boxes}
    if sum(b['class_id']==0 for b in boxes)!=1:raise ValueError('Need exactly one visible cassette')
    results=[b for b in boxes if b['class_id'] in (1,2)]
    rois=[b for b in boxes if b['class_id']==7]
    if not results or len(results)!=len(rois):raise ValueError('Need one ROI for each result window')
    for box in boxes:
        cid=box['class_id'];x0,y0,x1,y1=box['xyxy'];w,h=dimensions
        if cid not in (0,1,2,7) or not all(math.isfinite(x) for x in (x0,y0,x1,y1)):
            raise ValueError('Invalid class or nonfinite geometry')
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Box outside image: '+entry['image'])
        if cid==7:
            parent=by_id.get(box.get('outer_id'))
            if not parent or parent['class_id'] not in (1,2):raise ValueError('ROI missing exterior link')
            px0,py0,px1,py1=parent['xyxy']
            if not (px0<=x0<x1<=px1 and py0<=y0<y1<=py1):raise ValueError('ROI outside parent window')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('decisions',nargs='+');args=parser.parse_args()
    inventory=json.loads(INV.read_text(encoding='utf-8'));index={r['image']:r for r in inventory['images']}
    grouped=defaultdict(list)
    for row in inventory['images']:grouped[row['sha256']].append(row)
    accepted=[];excluded=[];reviews=[]
    for file in args.decisions:
        packet=json.loads(Path(file).read_text(encoding='utf-8-sig'))
        if not isinstance(packet,list):raise ValueError('Decisions must be an array')
        for entry in packet:
            if entry['image'] not in index:raise ValueError('Unknown image')
            validate(entry,ROOT/'images'/entry['image'])
            {'accepted':accepted,'excluded_overexposed':excluded,'needs_review':reviews}[entry['decision']].append(entry)
    documents={};locations={};hash_splits=defaultdict(set);session_splits=defaultdict(set)
    for split in ('train','val'):
        file=OUT/split/'annotations'/'annotations.json';doc=json.loads(file.read_text(encoding='utf-8'))
        if {c['id']:c['name'] for c in doc['categories']}!={i:n for i,n in enumerate(NAMES)}:
            raise ValueError('Eight-class schema mismatch')
        documents[split]=(file,doc)
        for image in doc['images']:
            name=image['file_name'];p=OUT/split/'images'/name
            if not p.is_file():raise ValueError('Missing output image')
            digest=sha(p);hash_splits[digest].add(split);session_splits[session(name)].add(split)
            if name in locations:raise ValueError('Duplicate filename across partitions')
            locations[name]=split
    if any(len(x)>1 for x in hash_splits.values()):raise ValueError('Identical photos cross partitions')
    plans={}; exclusion_plans={}
    for e in accepted:
        for row in grouped[e['sha256']]:
            name=row['image']
            if name in locations:continue
            if row['status'].startswith('excluded'):raise ValueError('Accepted decision contradicts exclusion: '+name)
            prior=plans.get(name)
            if prior and prior[0]['boxes']!=e['boxes']:raise ValueError('Conflicting visual boxes for identical image')
            plans[name]=(e,row)
    for e in excluded:
        for row in grouped[e['sha256']]:
            if row['image'] in plans:raise ValueError('Image accepted and excluded')
            exclusion_plans[row['image']]=(e,row)
    counts=Counter()
    for name,(e,row) in plans.items():
        src=ROOT/'images'/name
        if sha(src)!=e['sha256']:raise ValueError('Duplicate source changed')
        split_set=hash_splits.get(e['sha256'],set()) or session_splits.get(session(name),set())
        if len(split_set)==1:split=next(iter(split_set))
        else:
            ntrain=len(documents['train'][1]['images']);ntotal=ntrain+len(documents['val'][1]['images'])
            split='train' if ntrain<.7*(ntotal+1) else 'val'
        file,doc=documents[split];w,h=e['width'],e['height']
        iid=max((i['id'] for i in doc['images']),default=0)+1
        aid=max((a['id'] for a in doc['annotations']),default=0)+1
        doc['images'].append({'id':iid,'file_name':name,'width':w,'height':h})
        txt=[];box_ids={};links=[]
        for box in e['boxes']:
            x0,y0,x1,y1=box['xyxy'];bw=x1-x0;bh=y1-y0;cid=box['class_id'];box_ids[box['id']]=aid
            doc['annotations'].append({'id':aid,'image_id':iid,'category_id':cid,'bbox':[x0,y0,bw,bh],
                                      'area':bw*bh,'iscrowd':0})
            txt.append(f'{cid} {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} {bw/w:.8f} {bh/h:.8f}');aid+=1
        for box in e['boxes']:
            if box['class_id']==7:links.append({'roi_annotation_id':box_ids[box['id']],
                                               'outer_annotation_id':box_ids[box['outer_id']]})
        label=OUT/split/'labels'/f'{Path(name).stem}.txt'
        if label.exists() or (OUT/split/'images'/name).exists():raise ValueError('Unexpected output target exists')
        shutil.copy2(src,OUT/split/'images'/name);label.write_text('\n'.join(txt)+'\n',encoding='utf-8')
        row.update(status='completed_visual_review' if name==e['image'] else 'completed_identical_copy',
                   visual_validation=name==e['image'],identical_to=e['image'],split=split,
                   reviewed_label=str(label),roi_links=links,result_basis=e.get('reason',''))
        locations[name]=split;hash_splits[e['sha256']].add(split);session_splits[session(name)].add(split)
        counts['integrated']+=1
    for name,(e,row) in exclusion_plans.items():
        split=locations.get(name)
        if split:
            doc=documents[split][1];ids={im['id'] for im in doc['images'] if im['file_name']==name}
            doc['images']=[im for im in doc['images'] if im['id'] not in ids]
            doc['annotations']=[a for a in doc['annotations'] if a['image_id'] not in ids]
            for p in (OUT/split/'images'/name,OUT/split/'labels'/f'{Path(name).stem}.txt'):
                target=p.resolve();target.relative_to(OUT.resolve())
                if target.is_file():target.unlink()
            counts['removed_output_images']+=1
        row.update(status='excluded_overexposed',exclusion_reason=e.get('reason',''),visual_validation=name==e['image'])
        counts['excluded']+=1
    for e in reviews:
        for row in grouped[e['sha256']]:
            if row['image'] in locations:continue
            row.update(status='needs_review',visual_review=True,review_reason=e.get('reason',''),
                       review_proposals=e.get('proposals',e.get('boxes',[])),reviewed_source=e['image'])
            counts['needs_review']+=1
    for file,doc in documents.values():file.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    INV.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'changes':dict(counts),'train':len(documents['train'][1]['images']),
                      'val':len(documents['val'][1]['images'])},indent=2))

if __name__=='__main__':main()
