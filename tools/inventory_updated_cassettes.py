"""Inventory image/label integrity and flag exposure for visual review."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageStat

ROOT=Path(r'D:\Projects\IA\data_set')
OUT=ROOT/'revision_2075'
EXT={'.png','.jpg','.jpeg','.bmp','.webp'}

def main():
    OUT.mkdir(exist_ok=True)
    records=[]; hashes=defaultdict(list)
    for image in sorted(p for p in (ROOT/'images').iterdir() if p.suffix.lower() in EXT):
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        hashes[digest].append(image.name)
        label=ROOT/'labels'/f'{image.stem}.txt'
        with Image.open(image) as im:
            w,h=im.size
            gray=im.convert('L'); gray.thumbnail((256,256))
            hist=gray.histogram(); total=sum(hist)
            white=sum(hist[245:])/total; black=sum(hist[:11])/total
            avg=ImageStat.Stat(gray).mean[0]
        rows=[]; errors=[]
        if label.exists():
            for n,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
                if not line.strip(): continue
                try:
                    values=line.split()
                    if len(values)!=5: raise ValueError('expected 5 values')
                    c,x,y,bw,bh=map(float,values)
                    if c!=int(c) or not 0<=c<=7 or bw<=0 or bh<=0 or any(v<0 or v>1 for v in (x,y,bw,bh)):
                        raise ValueError('invalid class or coordinates')
                    if x-bw/2 < -1e-6 or y-bh/2 < -1e-6 or x+bw/2 > 1+1e-6 or y+bh/2 > 1+1e-6:
                        errors.append(f'{n}: box exceeds image')
                    rows.append(int(c))
                except ValueError as e: errors.append(f'{n}: {e}')
        records.append({'image':image.name,'width':w,'height':h,'sha256':digest,
                        'label_exists':label.exists(),'label_count':len(rows),'class_counts':dict(Counter(rows)),
                        'label_errors':errors,'mean_luminance':round(avg,2),
                        'near_white_fraction':round(white,4),'near_black_fraction':round(black,4),
                        'status':'needs_visual_review',
                        'exposure_flag':'review_bright' if white>=0.45 else ('review_dark' if black>=0.45 else 'not_assessed')})
    summary={'image_count':len(records),'label_match':sum(r['label_exists'] for r in records),
             'nonempty_label_match':sum(r['label_count']>0 for r in records),
             'label_errors':sum(bool(r['label_errors']) for r in records),
             'bright_candidates':sum(r['exposure_flag']=='review_bright' for r in records),
             'dark_candidates':sum(r['exposure_flag']=='review_dark' for r in records),
             'unique_hashes':len(hashes),'duplicate_groups':sum(len(v)>1 for v in hashes.values()),
             'exposure_note':'Brightness flags are screening signals only; actual exclusions require visual review.',
             'images':records}
    (OUT/'inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='images'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
