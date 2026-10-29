"""Validate recovered visual records and collect them without repeating vision calls."""
import json
from collections import Counter
from pathlib import Path
from integrate_cassette_visual_decisions import validate,ROOT

QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
records=[];problems=[];names=set()
for number in range(1,7):
    file=QA/f'cassette_worker_{number}_recovered.json'
    if not file.exists():continue
    for row in json.loads(file.read_text(encoding='ascii')):
        if row['image'] in names:raise ValueError('Duplicate worker assignments')
        names.add(row['image'])
        try:validate(row,ROOT/'images'/row['image'])
        except (ValueError,KeyError,TypeError) as exc:
            problems.append({'image':row['image'],'error':str(exc)})
            row['proposals']=row.get('boxes',[])
            row.update(decision='needs_review',boxes=[],reason='Checkpoint geometry requires correction: '+str(exc))
        records.append(row)
dest=QA/'cassette_recovered_validated.json'
dest.write_text(json.dumps(records,ensure_ascii=True,indent=2),encoding='ascii')
(QA/'cassette_recovery_validation_errors.json').write_text(json.dumps(problems,indent=2),encoding='ascii')
print(json.dumps({'records':len(records),'counts':dict(Counter(x['decision'] for x in records)),
                  'invalid_geometry':len(problems),'examples':problems[:5]},indent=2))
