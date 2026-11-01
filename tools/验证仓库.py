"""Validate indexed files, README links, archives, and current MuJoCo assets."""
from pathlib import Path
from urllib.parse import quote,unquote
import re
import sys
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parent.parent
skip={'.git','.venv','.pio','__pycache__','warp_cache','gpu_packages','warp_test_packages'}
archives={'.zip','.7z','.rar','.tar','.gz','.tgz','.bz2','.xz'}
files=[x for x in root.rglob('*') if x.is_file() and not any(s in skip for s in x.relative_to(root).parts)]
bad=[x.relative_to(root).as_posix() for x in files if x.suffix.lower() in archives]
index=(root/'文件清单.md').read_text(encoding='utf8')
missing=[x.relative_to(root).as_posix() for x in files if f']({quote(x.relative_to(root).as_posix(),safe="/")})' not in index]
readme=(root/'README.md').read_text(encoding='utf8')
links=re.findall(r'\]\(([^)]+)\)',readme)
broken=[]
for url in links:
 if url.startswith(('https://','http://','#','mailto:')):continue
 if not (root/unquote(url.split('#')[0])).exists():broken.append(url)
candidate=root/'simulation'/'diagnostic_post_shaft_fix_gear_phase_25.xml'
model=ET.parse(candidate).getroot()
assets=[]
for element in model.findall('./asset/*'):
 f=element.get('file')
 if f and not (candidate.parent/f).is_file():assets.append(f)
print('file count',len(files),'unindexed',len(missing),'archive files',len(bad),
      'broken README links',len(broken),'missing current-model assets',len(assets))
for label,items in [('unindexed',missing),('archives',bad),('broken links',broken),('missing assets',assets)]:
 if items:print(label,*items[:10],sep='\n')
if missing or bad or broken or assets:sys.exit(1)
