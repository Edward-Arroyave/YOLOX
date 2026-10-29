"""Restore original class IDs to the already exported reviewed dataset."""
import json
from pathlib import Path

ROOT = Path(r'D:\Projects\IA\data_set\yolox_revisado')
NAMES = ['cassette','negative','positive','dual_test_negative',
         'dual_test_2_positive','dual_test_1_positive','dual_test_both_positive','ROI']
COLORS = ['#00e1ff','#04f500','#ff0000','#24ff6d','#ff6b35','#ff9f1c','#ff0019','#ff0019']

def main():
    pending=[]
    for split in ('train','val'):
        base=ROOT/split
        coco_path=base/'annotations'/'annotations.json'
        coco=json.loads(coco_path.read_text(encoding='utf-8'))
        if {c['id']:c['name'] for c in coco['categories']} != {0:'cassette',1:'negative',2:'positive',7:'ROI'}:
            raise ValueError(f'Unexpected categories in {coco_path}; stopped before edits')
        for ann in coco['annotations']:
            if ann['category_id'] not in (0,1,2,7): raise ValueError('Unexpected annotation class')
        files=[]
        for f in (base/'labels').glob('*.txt'):
            rows=[]
            for line in f.read_text(encoding='utf-8').splitlines():
                p=line.split()
                if len(p)!=5 or p[0] not in ('0','1','2','3'): raise ValueError(f'Unexpected TXT row in {f}')
                if p[0]=='3': p[0]='7'
                rows.append(' '.join(p))
            files.append((f,'\n'.join(rows)+'\n'))
        coco['categories']=[{'id':i,'name':name,'supercategory':'none','color':COLORS[i]} for i,name in enumerate(NAMES)]
        yaml=(f'path: {ROOT.as_posix()}\ntrain: train/images\nval: val/images\nnc: 8\nnames:\n'+
              ''.join(f'  "{i}": {name}\n' for i,name in enumerate(NAMES))+
              'colors:\n'+''.join(f'  "{i}": "{color}"\n' for i,color in enumerate(COLORS)))
        pending.append((coco_path,json.dumps(coco,indent=2)))
        pending.append((base/'data.yaml',yaml))
        pending.extend(files)
    for path,content in pending: path.write_text(content,encoding='utf-8')
    print(f'updated_files={len(pending)}; categories=8; ROI=7')

if __name__=='__main__': main()
