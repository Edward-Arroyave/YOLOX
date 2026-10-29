import json
from pathlib import Path
from collections import defaultdict

ROOT=Path(r'D:\Projects\IA\data_set')
QA=Path(r'C:\Users\Edward.Arrollave\.codex\visualizations\2026\10\29\01a185e3-f434-7e60-b941-9c824cf7a864')
inv=json.loads((ROOT/'revision_2075'/'inventory.json').read_text(encoding='utf-8'))
pending=[r for r in inv['images'] if r['status']=='needs_visual_review']
reserved=pending[:20]
by_session=defaultdict(list)
seen={r['sha256'] for r in reserved}
for row in pending[20:]:
    if row['sha256'] in seen: continue
    seen.add(row['sha256'])
    key=row['image'].rsplit('_',1)[0]
    by_session[key].append(row)
shards=[[] for _ in range(6)]
for group in by_session.values():
    shard=min(range(6),key=lambda i:len(shards[i])); shards[shard].extend(group)
for i,rows in enumerate(shards):
    dest=QA/f'cassette_worker_{i+1}_queue.json'
    dest.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{i+1}: {len(rows)} images: {dest}')
(QA/'cassette_parent_queue.json').write_text(json.dumps(reserved,indent=2),encoding='utf-8')
