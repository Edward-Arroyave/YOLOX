"""Reuse a reviewed annotation only for files with identical SHA-256 bytes."""
import json
from collections import defaultdict, Counter
from pathlib import Path
import shutil

ROOT = Path(r'D:\Projects\IA\data_set')
PILOT = ROOT/'revision_visual_piloto'
RESUME = ROOT/'revision_reanudable'

def main():
    path = RESUME/'review.json'
    review = json.loads(path.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for row in review['images']: groups[row['sha256']].append(row)
    coco_path = PILOT/'annotations'/'annotations.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    links_path = PILOT/'roi_links.json'
    links = json.loads(links_path.read_text(encoding='utf-8'))
    image_index = {x['file_name']:x for x in coco['images']}
    next_image = max(x['id'] for x in coco['images'])+1
    next_ann = max(x['id'] for x in coco['annotations'])+1
    changes=[]
    for digest, group in groups.items():
        sources=[r for r in group if r['status']=='completed' and r['image'] in image_index]
        if not sources: continue
        source=sources[0]
        source_image=image_index[source['image']]
        source_annotations=[a for a in coco['annotations'] if a['image_id']==source_image['id']]
        source_links=[link for link in links if link['roi_annotation_id'] in {a['id'] for a in source_annotations}]
        source_label=PILOT/'labels'/Path(source['image']).with_suffix('.txt').name
        for row in group:
            if row['status']!='needs_review' or row['image'] in image_index: continue
            target_image=next_image; next_image+=1
            coco['images'].append({'id':target_image,'file_name':row['image'],'width':source_image['width'],'height':source_image['height']})
            mapping={}
            for ann in source_annotations:
                new=dict(ann); mapping[ann['id']]=next_ann
                new.update(id=next_ann,image_id=target_image); next_ann+=1
                coco['annotations'].append(new)
            for link in source_links:
                links.append({'roi_annotation_id':mapping[link['roi_annotation_id']], 'outer_annotation_id':mapping[link['outer_annotation_id']]})
            target_label=PILOT/'labels'/Path(row['image']).with_suffix('.txt').name
            shutil.copy2(source_label,target_label)
            shutil.copy2(PILOT/'previews'/source['image'],PILOT/'previews'/row['image'])
            row.update(status='completed_identical_copy',visual_validation=False, reviewed_label=str(target_label),
                       identical_to=source['image'], evidence='SHA-256 igual; imagen y dimensiones idénticas.')
            changes.append({'image':row['image'],'status':'completed_identical_copy','identical_to':source['image']})
    coco_path.write_text(json.dumps(coco,indent=2),encoding='utf-8')
    links_path.write_text(json.dumps(links,indent=2),encoding='utf-8')
    path.write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    with (PILOT/'changes.jsonl').open('a',encoding='utf-8') as f:
        for row in changes: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'copied':len(changes),'statuses':dict(Counter(r['status'] for r in review['images']))},indent=2))

if __name__=='__main__': main()
