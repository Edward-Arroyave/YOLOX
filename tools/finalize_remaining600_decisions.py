"""Apply final worker corrections after all 600 visual records are complete."""
import json
from collections import defaultdict,Counter
from pathlib import Path
from integrate_cassette_visual_decisions import ROOT,OUT,INV,validate

QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')

def main():
    entries=[]
    for number in range(1,5):
        queue=json.loads((QA/f'remaining600_{number}_queue.json').read_text(encoding='ascii'))
        rows=json.loads((QA/f'remaining600_{number}_decisions.json').read_text(encoding='ascii'))
        if {x['image'] for x in queue}!={x['image'] for x in rows}:raise ValueError(f'Worker {number} is incomplete')
        entries.extend(rows)
    if len(entries)!=600 or len({e['image'] for e in entries})!=600:raise ValueError('Expected 600 unique final records')
    for e in entries:validate(e,ROOT/'images'/e['image'])
    documents={};locations={}
    for split in ('train','val'):
        path=OUT/split/'annotations'/'annotations.json';doc=json.loads(path.read_text(encoding='utf-8'));documents[split]=(path,doc)
        for im in doc['images']:locations[im['file_name']]=(split,im)
    inventory=json.loads(INV.read_text(encoding='utf-8'));index={r['image']:r for r in inventory['images']}
    changes=Counter()
    for e in entries:
        loc=locations.get(e['image'])
        if e['decision']=='accepted' and not loc:raise ValueError('Final accepted image not integrated: '+e['image'])
    for e in entries:
        name=e['image'];loc=locations.get(name)
        if e['decision']!='accepted':
            if loc:
                split,im=loc;doc=documents[split][1]
                doc['images']=[x for x in doc['images'] if x['id']!=im['id']]
                doc['annotations']=[x for x in doc['annotations'] if x['image_id']!=im['id']]
                for target in (OUT/split/'images'/name,OUT/split/'labels'/f'{Path(name).stem}.txt'):
                    target.resolve().relative_to(OUT.resolve())
                    if target.is_file():target.unlink()
                changes['withdrawn_uncertain']+=1
            continue
        split,im=loc;doc=documents[split][1];w,h=e['width'],e['height'];lines=[]
        for box in e['boxes']:
            x0,y0,x1,y1=box['xyxy'];lines.append(f'{box["class_id"]} {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} {(x1-x0)/w:.8f} {(y1-y0)/h:.8f}')
        expected='\n'.join(lines)+'\n';label=OUT/split/'labels'/f'{Path(name).stem}.txt'
        if label.read_text(encoding='utf-8')==expected:continue
        previous=[a['id'] for a in doc['annotations'] if a['image_id']==im['id']];nextid=max(a['id'] for a in doc['annotations'])+1
        doc['annotations']=[a for a in doc['annotations'] if a['image_id']!=im['id']];ids={};links=[]
        for i,b in enumerate(e['boxes']):
            aid=previous[i] if i<len(previous) else nextid
            if i>=len(previous):nextid+=1
            x0,y0,x1,y1=b['xyxy'];bw=x1-x0;bh=y1-y0;ids[b['id']]=aid
            doc['annotations'].append({'id':aid,'image_id':im['id'],'category_id':b['class_id'],'bbox':[x0,y0,bw,bh],'area':bw*bh,'iscrowd':0})
        for b in e['boxes']:
            if b['class_id']==7:links.append({'roi_annotation_id':ids[b['id']],'outer_annotation_id':ids[b['outer_id']]})
        index[name]['roi_links']=links;label.write_text(expected,encoding='utf-8');changes['corrected_final_geometry']+=1
    for path,doc in documents.values():path.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    INV.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'final_records':len(entries),'decisions':dict(Counter(e['decision'] for e in entries)),'changes':dict(changes)},indent=2))

if __name__=='__main__':main()
