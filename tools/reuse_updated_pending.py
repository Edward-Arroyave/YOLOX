"""Transfer previous provisional reviews only for byte-identical photos."""
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(r'D:\Projects\IA\data_set')
NEW=ROOT/'revision_2075'
OLD=ROOT/'revision_reanudable'

def main():
    invpath=NEW/'inventory.json'; inv=json.loads(invpath.read_text(encoding='utf-8'))
    old=json.loads((OLD/'review.json').read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for row in old['images']: groups[row['sha256']].append(row)
    folder=NEW/'propuestas'; folder.mkdir(exist_ok=True)
    counts=Counter()
    for row in inv['images']:
        if row['status']!='needs_visual_review': continue
        matches=groups.get(row['sha256'],[])
        candidate=next((x for x in matches if x['status']=='needs_review_protocol_and_geometry'),None)
        excluded=next((x for x in matches if x['status']=='excluded_overexposed'),None)
        if candidate:
            proposal=json.loads(Path(candidate['proposal']).read_text(encoding='utf-8'))
            proposal.update(image=row['image'],identical_to=candidate['image'],sha256=row['sha256'])
            dest=folder/(Path(row['image']).stem+'.json')
            dest.write_text(json.dumps(proposal,ensure_ascii=False,indent=2),encoding='utf-8')
            row.update(status='needs_review_protocol_and_geometry',proposal=str(dest),identical_to=candidate['image'])
            counts['proposals']+=1
        elif excluded:
            row.update(status='excluded_overexposed_identical',identical_to=excluded['image'],
                       exclusion_reason='Imagen idéntica a una fotografía excluida tras revisión visual.')
            counts['excluded']+=1
    invpath.write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf-8')
    print(dict(counts))

if __name__=='__main__':main()
