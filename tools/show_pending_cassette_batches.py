"""Compose copies of five originals per sheet for visual review; never change source images."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
inv=json.loads((ROOT/'revision_2075'/'inventory.json').read_text(encoding='utf-8'))
pending=[r for r in inv['images'] if r['status']=='needs_visual_review'][:20]
for start in range(0,len(pending),5):
    group=pending[start:start+5]; width=sum(r['width'] for r in group)
    sheet=Image.new('RGB',(width,1000),'#202020'); draw=ImageDraw.Draw(sheet); x=0
    for i,r in enumerate(group):
        with Image.open(ROOT/'images'/r['image']) as im: sheet.paste(im.convert('RGB'),(x,40))
        draw.text((x+5,3),f'{start+i}: {r["image"].split("medical_order_")[-1]}',fill='white')
        x+=r['width']
    dest=QA/f'pending_sheet_{start//5}.png'; sheet.save(dest); print(dest)
(QA/'pending_sheets.json').write_text(json.dumps(pending,indent=2),encoding='utf-8')
