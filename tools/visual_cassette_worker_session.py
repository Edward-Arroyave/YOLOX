"""Small shared helper for human/model visual review checkpoints, never a label predictor."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

SOURCE=Path(r'D:\Projects\IA\data_set\images')

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sheet',type=int);p.add_argument('--overlay',action='store_true');p.add_argument('--spec')
    a=p.parse_args();output=Path(a.output);queue=json.loads(Path(a.queue).read_text(encoding='ascii'))
    existing=json.loads(output.read_text(encoding='ascii')) if output.exists() else []
    saved={x['image']:x for x in existing}
    if a.spec:
        spec=json.loads(Path(a.spec).read_text(encoding='ascii'))
        for row in spec:
            q=queue[row['index']]
            if hashlib.sha256((SOURCE/q['image']).read_bytes()).hexdigest()!=q['sha256']:raise ValueError('Source hash mismatch')
            e={k:q[k] for k in ('image','sha256','width','height')};boxes=[]
            if row.get('cassette'):boxes.append({'id':'cassette','class_id':0,'xyxy':row['cassette']})
            for n,(c,outer,roi) in enumerate(row.get('windows',[]),1):
                if c not in (1,2):raise ValueError('Uncertain windows belong in proposals')
                boxes.extend([{'id':f'window_{n}','class_id':c,'xyxy':outer},
                              {'id':f'roi_{n}','class_id':7,'xyxy':roi,'outer_id':f'window_{n}'}])
            for b in boxes:
                x0,y0,x1,y1=b['xyxy']
                if not 0<=x0<x1<=q['width'] or not 0<=y0<y1<=q['height']:raise ValueError('Invalid bounds')
            e.update(decision=row['decision'],visually_reviewed=True,boxes=boxes,reason=row.get('reason','')[:180])
            if row.get('proposals'):e['proposals']=row['proposals']
            saved[e['image']]=e
        rows=[saved[x['image']] for x in queue if x['image'] in saved]
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(rows,ensure_ascii=True,indent=2),encoding='ascii');temp.replace(output)
        print('saved',len(rows),flush=True)
    if a.sheet is not None:
        group=queue[a.sheet:a.sheet+5];sheet=Image.new('RGB',(sum(x['width']+16 for x in group),1020),'#dddddd')
        draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);offset=0
        for i,q in enumerate(group,a.sheet):
            with Image.open(SOURCE/q['image']) as original:im=original.convert('RGB')
            d=ImageDraw.Draw(im)
            if a.overlay:
                for b in saved.get(q['image'],{}).get('boxes',[]):
                    d.rectangle(b['xyxy'],outline={0:'cyan',1:'lime',2:'red',7:'yellow'}[b['class_id']],width=2)
            sheet.paste(im,(offset,35));draw.text((offset+2,2),f'#{i} {q["width"]}x{q["height"]}',fill='black',font=font)
            for y in range(0,q['height'],100):draw.text((offset+2,y+36),str(y),fill='#0077cc',font=font)
            for x in range(100,q['width'],100):draw.text((offset+x,990),str(x),fill='#0077cc',font=font)
            offset+=q['width']+16
        dest=output.parent/f'{output.stem}_{"overlay" if a.overlay else "sheet"}_{a.sheet:03d}.jpg'
        sheet.save(dest,quality=93);print(dest,flush=True)

if __name__=='__main__':main()
