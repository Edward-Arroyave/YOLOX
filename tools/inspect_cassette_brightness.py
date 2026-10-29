"""Rank exposure candidates for visual inspection, without discarding images."""
import json
from pathlib import Path
from PIL import Image, ImageStat

ROOT = Path(r'D:\Projects\IA\data_set')
OUT = ROOT/'revision_reanudable'/'exposure_candidates.json'
items = []
for path in sorted((ROOT/'images').glob('*.png')):
    with Image.open(path) as im:
        gray = im.convert('L')
        hist = gray.histogram()
        pixels = sum(hist)
        white = sum(hist[245:])/pixels
        mean = ImageStat.Stat(gray).mean[0]
        items.append({'image':path.name,'mean_luminance':round(mean,2),
                      'fraction_near_white':round(white,5),
                      'decision':'pending_visual_inspection'})
items.sort(key=lambda item:(item['fraction_near_white'],item['mean_luminance']), reverse=True)
OUT.write_text(json.dumps({'description':'Exposure ranking only; brightness alone does not establish unreadability.',
                           'images':items},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(items[:10],ensure_ascii=False,indent=2))
