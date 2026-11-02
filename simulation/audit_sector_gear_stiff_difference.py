"""Assert only two intended gear solver parameters differ from baseline."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
p=Path(__file__).parent;a=ET.parse(p/'diagnostic_sector_collision_trial.xml').getroot();b=ET.parse(p/'diagnostic_sector_gear_stiff_trial.xml').getroot();changes=[]
for x,y in zip(a.iter(),b.iter()):
 if x.tag!=y.tag or len(x)!=len(y):raise RuntimeError('XML structure differs')
 ax=dict(x.attrib);ay=dict(y.attrib)
 if x is a:ax.pop('model',None);ay.pop('model',None)
 if x.tag=='joint' and x.get('name') in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50'):
  changes.append({'name':x.get('name'),'old':ax.pop('solref',None),'new':ay.pop('solref',None)})
 if ax!=ay or (x.text or '').strip()!=(y.text or '').strip():raise RuntimeError('Unexpected difference')
if len(changes)!=2 or any(x['old']!='0.001 1' or x['new']!='0.0005 1' for x in changes):raise RuntimeError('Gear changes not as expected')
out={'verified':True,'changes':changes,'limitations':'Only XML comparison; no stability or hardware guarantee.'}
(p/'sector_gear_stiff_difference_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out)
