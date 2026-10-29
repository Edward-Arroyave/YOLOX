"""Record manually inspected simple/double cassette images with visible C/T marks."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(r'D:\Projects\IA\data_set')
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
queue=json.loads((QA/'pending_sheets.json').read_text(encoding='utf-8'))
rois={
0:[[119,277,151,415]],1:[[120,278,150,415]],2:[[122,275,153,413]],3:[[120,280,151,415]],4:[[120,275,152,413]],
5:[[112,321,161,540]],6:[[113,320,161,539]],7:[[113,317,161,539]],8:[[112,319,160,539]],9:[[113,319,161,539]],
10:[[174,406,202,537]],11:[[176,403,203,538]],12:[[175,406,204,541]],13:[[172,403,202,536]],14:[[175,405,204,538]],
15:[[108,351,153,564],[309,357,354,570]],16:[[107,353,153,568],[308,355,353,568]],
17:[[108,352,153,566],[308,354,353,567]],18:[[105,350,151,564],[308,353,352,566]],
19:[[109,348,154,563],[309,352,354,567]],
}
decisions=[]
for i,row in enumerate(queue):
    entry={k:row[k] for k in ('image','sha256','width','height')}
    if i<5:
        cassette=[32,57,235,699];outside=[[46,251,214,443]];classes=[1]
    elif i<10:
        cassette=[0,0,280,960];outside=[[8,304,274,577]];classes=[2]
    elif i<15:
        cassette=[94,195,281,825];outside=[[105,382,266,567]];classes=[1]
    else:
        cassette=[0,0,464,960];outside=[[49,315,267,616],[207,319,418,617]];classes=[1,2]
    boxes=[{'id':'cassette','class_id':0,'xyxy':cassette}]
    for n,(cls,outer,roi) in enumerate(zip(classes,outside,rois[i]),1):
        boxes.extend([{'id':f'window_{n}','class_id':cls,'xyxy':outer},
                      {'id':f'roi_{n}','class_id':7,'xyxy':roi,'outer_id':f'window_{n}'}])
    entry.update(decision='accepted',visually_reviewed=True,boxes=boxes,
                 reason='Marcas C/T visibles; control localizado. Resultado asignado por presencia de test en la posición T; ROI sobre toda la tira visible.')
    decisions.append(entry)
for start in range(0,20,5):
    group=decisions[start:start+5];sheet=Image.new('RGB',(sum(r['width'] for r in group),1000),'#202020');draw=ImageDraw.Draw(sheet);offset=0
    for r in group:
        with Image.open(ROOT/'images'/r['image']) as im:preview=im.convert('RGB')
        d=ImageDraw.Draw(preview)
        for b in r['boxes']:d.rectangle(b['xyxy'],outline={0:'cyan',1:'lime',2:'red',7:'yellow'}[b['class_id']],width=2)
        sheet.paste(preview,(offset,40));draw.text((offset+2,2),r['image'].split('medical_order_')[-1],fill='white');offset+=r['width']
    path=QA/f'next_twenty_overlay_{start//5}.png';sheet.save(path);print(path)
(QA/'next_twenty_decisions.json').write_text(json.dumps(decisions,ensure_ascii=True,indent=2),encoding='ascii')
