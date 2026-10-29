import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(r'D:\Projects\IA\data_set')
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
inventory=json.loads((ROOT/'revision_2075'/'inventory.json').read_text(encoding='utf-8'))
groups=defaultdict(list)
for row in inventory['images']:
    if row['status']=='needs_visual_review':groups[row['image'].rsplit('_',1)[0]].append(row)
shards=[[] for _ in range(4)]
for group in groups.values():shards[min(range(4),key=lambda n:len(shards[n]))].extend(group)
for i,shard in enumerate(shards,1):
    dest=QA/f'remaining600_{i}_queue.json';dest.write_text(json.dumps(shard,ensure_ascii=True,indent=2),encoding='ascii')
    print(i,len(shard),dest)
