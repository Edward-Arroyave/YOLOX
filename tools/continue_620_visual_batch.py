import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
queue=json.loads((QA/'pending_sheets.json').read_text(encoding='utf-8'))
rois={
5:[[116,373,147,510]],6:[[116,370,148,511]],7:[[115,369,145,510]],8:[[115,371,146,510]],9:[[116,370,147,510]],
10:[[92,264,149,525],[334,263,390,522]],11:[[94,274,151,525],[329,271,382,522]],
12:[[94,265,148,526],[335,267,390,526]],13:[[97,264,151,521],[338,267,392,523]],14:[[97,262,152,524],[342,266,394,526]],
15:[[95,389,143,598],[294,385,341,596]],16:[[94,384,142,596],[295,384,342,596]],
17:[[95,389,143,599],[295,386,341,596]],18:[[99,386,145,595],[299,386,343,595]],19:[[94,383,141,597],[295,382,339,596]],
}
decisions=[]
for i,row in enumerate(queue):
    e={k:row[k] for k in ('image','sha256','width','height')};e['visually_reviewed']=True
    if i<5:
        e.update(decision='needs_review',boxes=[],reason='Desenfoque intenso impide localizar marcas C/T y delimitar con certeza la tira; aparece un segundo casete parcialmente cortado en la parte inferior.')
    else:
        if i<10:
            cassette=[34,145,229,824];outers=[[43,345,215,541]];classes=[1]
        elif i<15:
            cassette=[0,0,464,960];outers=[[17,218,276,579],[215,218,463,579]];classes=[1,2]
        else:
            cassette=[0,0,464,960];outers=[[34,347,254,643],[194,347,398,643]];classes=[1,1]
        boxes=[{'id':'cassette','class_id':0,'xyxy':cassette}]
        for n,(c,outer,roi) in enumerate(zip(classes,outers,rois[i]),1):
            boxes.extend([{'id':f'window_{n}','class_id':c,'xyxy':outer},
                          {'id':f'roi_{n}','class_id':7,'xyxy':roi,'outer_id':f'window_{n}'}])
        e.update(decision='accepted',boxes=boxes,reason='C/T localizadas visualmente; C visible. Clase por presencia de T en su posicion; ROI medida sobre la tira completa.')
    decisions.append(e)
for start in range(0,20,5):
    group=decisions[start:start+5];sheet=Image.new('RGB',(sum(r['width'] for r in group),1000),'#202020');d=ImageDraw.Draw(sheet);x=0
    for r in group:
        with Image.open(ROOT/'images'/r['image']) as im:pic=im.convert('RGB')
        draw=ImageDraw.Draw(pic)
        for b in r['boxes']:draw.rectangle(b['xyxy'],outline={0:'cyan',1:'lime',2:'red',7:'yellow'}[b['class_id']],width=2)
        sheet.paste(pic,(x,40));d.text((x+2,2),r['decision'],fill='white');x+=r['width']
    p=QA/f'continue620_overlay_{start//5}.png';sheet.save(p);print(p)
(QA/'continue620_batch1_decisions.json').write_text(json.dumps(decisions,ensure_ascii=True,indent=2),encoding='ascii')
