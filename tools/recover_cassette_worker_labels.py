"""Recover bounded ASCII decision records from recursively mojibaked checkpoints."""
import json
from collections import Counter
from pathlib import Path

QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
NONASCII=bytes(range(128,256))

def small(value):
    if isinstance(value,str): return value[:600]
    if isinstance(value,list):return [small(v) for v in value[:20]]
    if isinstance(value,dict):return {str(k)[:60]:small(v) for k,v in value.items() if k not in ('pixels','image_data','base64')}
    return value

def main():
    for number in range(1,7):
        source=QA/f'cassette_worker_{number}_decisions.json'
        if not source.exists():continue
        compact=QA/f'cassette_worker_{number}_recovered.json'
        temporary=QA/f'worker{number}_ascii.tmp'
        # Drop non-ASCII bytes in chunks: names, hashes, keys and geometry are ASCII.
        # Corrupted accented reason text is non-authoritative and never used to assign classes.
        with source.open('rb') as inp,temporary.open('wb') as out:
            while chunk:=inp.read(8*1024*1024):out.write(chunk.translate(None,NONASCII))
        data=json.loads(temporary.read_text(encoding='ascii'))
        recovered=[]
        for row in data:
            clean={k:row[k] for k in ('image','sha256','width','height','decision','visually_reviewed')}
            clean['boxes']=[{k:box[k] for k in ('id','class_id','xyxy','outer_id') if k in box} for box in row.get('boxes',[])]
            clean['reason']=str(row.get('reason',''))[:600]
            if 'proposals' in row:clean['proposals']=small(row['proposals'])
            recovered.append(clean)
        compact.write_text(json.dumps(recovered,ensure_ascii=True,indent=2),encoding='ascii')
        # Temporary bytes are a generated intermediate, recoverable from the original.
        temporary.unlink()
        print(json.dumps({'worker':number,'records':len(recovered),'counts':dict(Counter(r['decision'] for r in recovered)),
                          'recovered_bytes':compact.stat().st_size}),flush=True)

if __name__=='__main__':main()
