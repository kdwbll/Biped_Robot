import json, math
from collections import defaultdict
from pathlib import Path
base=Path(__file__).parent
mates=json.loads((base/'mates.json').read_text(encoding='utf8'))
groups=defaultdict(list)
for m in mates:
 if m['type'] != 'MateConcentric' or len(m['entities'])!=2: continue
 a,b=m['entities']; p=a['parameters']; q=b['parameters']
 if len(p)<6 or len(q)<6: continue
 # Difference along the cylinder axis is expected: test perpendicular axis alignment.
 delta=[q[i]-p[i] for i in range(3)]; axis=p[3:6]; dot=sum(delta[i]*axis[i] for i in range(3)); offset=math.sqrt(sum((delta[i]-dot*axis[i])**2 for i in range(3)))
 groups[m['assembly']].append({'mate':m['name'],'components':[a['component'],b['component']],'axis_point_m':p[:3],'axis_direction':axis,'axis_offset_m':offset})
result={k:v for k,v in groups.items()}
(base/'axis_candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for k,v in result.items(): print(k,'axes',len(v),'max mismatch mm',round(max(x['axis_offset_m'] for x in v)*1000,6))
