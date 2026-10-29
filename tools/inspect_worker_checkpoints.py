from pathlib import Path
import mmap

QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
for number in (2,5):
    file=QA/f'cassette_worker_{number}_decisions.json'
    print(file.name,file.stat().st_size)
    with file.open('rb') as stream:
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            for offset in (0,len(data)//2,max(0,len(data)-400)):
                print(offset,repr(data[offset:offset+200]))
