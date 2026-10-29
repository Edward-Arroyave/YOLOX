"""Record the parent agent's individually inspected twenty-image batch."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
ROOT=Path(r'D:\Projects\IA\data_set')
queue=json.loads((QA/'cassette_parent_queue.json').read_text(encoding='utf-8'))
# Left ROI, right ROI, and measured exterior bounds, in source-image pixels.
coords={
1:([94,326,141,540],[297,323,343,537],[42,292,251,583],[196,289,403,583]),
2:([93,326,141,540],[298,323,343,537],[42,292,251,583],[196,289,403,583]),
4:([93,328,140,541],[296,324,341,538],[41,293,250,585],[194,291,402,585]),
5:([107,366,156,589],[316,372,365,591],[49,331,274,638],[212,335,432,638]),
6:([107,367,156,585],[313,371,361,587],[49,332,274,636],[212,335,431,637]),
7:([110,369,158,584],[315,371,363,588],[51,334,275,635],[213,335,431,636]),
8:([112,373,159,588],[314,373,362,590],[52,335,275,637],[212,336,431,638]),
9:([110,371,158,586],[315,373,363,587],[50,333,275,636],[212,335,432,637]),
10:([116,358,163,569],[314,357,359,566],[60,323,272,614],[214,322,419,614]),
11:([116,353,162,568],[315,359,361,573],[59,319,274,616],[214,324,421,616]),
12:([114,357,161,568],[312,358,358,568],[59,322,272,614],[211,323,420,614]),
13:([111,354,158,569],[313,356,359,569],[56,320,272,614],[211,322,420,614]),
14:([114,355,161,571],[316,358,363,574],[59,321,276,619],[213,324,424,619]),
15:([114,343,163,563],[322,343,369,563],[56,307,280,610],[214,305,433,610]),
16:([112,342,161,562],[319,343,367,564],[54,306,278,611],[212,306,432,611]),
17:([110,340,159,563],[321,340,369,562],[53,304,278,610],[213,303,433,610]),
18:([113,339,161,559],[318,344,367,564],[55,304,278,612],[211,307,433,612]),
19:([113,344,163,560],[320,346,369,563],[55,308,279,611],[213,309,434,611]),
}
decisions=[]
for i,row in enumerate(queue):
    entry={k:row[k] for k in ('image','sha256','width','height')}
    entry['visually_reviewed']=True
    if i in (0,3):
        entry.update(decision='excluded_overexposed',boxes=[],reason='Sobreexposición impide distinguir con suficiente claridad las marcas y el marco de las ventanas.')
    else:
        left,right,outleft,outright=coords[i]
        cassette=[0,0,row['width'],row['height']]
        entry.update(decision='accepted',reason='Casete doble: marcas C/T compartidas, control visible en ambas tiras y test visible solo en derecha.',
          boxes=[{'id':'cassette','class_id':0,'xyxy':cassette},
                 {'id':'left','class_id':1,'xyxy':outleft},
                 {'id':'right','class_id':2,'xyxy':outright},
                 {'id':'roi_left','class_id':7,'xyxy':left,'outer_id':'left'},
                 {'id':'roi_right','class_id':7,'xyxy':right,'outer_id':'right'}])
    decisions.append(entry)
for start in range(0,len(decisions),5):
    group=decisions[start:start+5]; sheet=Image.new('RGB',(sum(x['width'] for x in group),1000),'#202020'); draw=ImageDraw.Draw(sheet); offset=0
    for r in group:
        with Image.open(ROOT/'images'/r['image']) as im: preview=im.convert('RGB')
        d=ImageDraw.Draw(preview)
        for box in r['boxes']:
            color={0:'cyan',1:'lime',2:'red',7:'yellow'}[box['class_id']]
            d.rectangle(box['xyxy'],outline=color,width=2)
        sheet.paste(preview,(offset,40)); draw.text((offset+3,2),r['decision'],fill='white');offset+=r['width']
    dest=QA/f'parent_overlay_{start//5}.png'; sheet.save(dest);print(dest)
(QA/'cassette_parent_decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
