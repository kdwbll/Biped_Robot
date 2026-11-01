import json,numpy as np,win32com.client as w
from collections import defaultdict
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim');s=w.GetActiveObject('SldWorks.Application.33'); main=next(d for d in s.GetDocuments if d.GetTitle=='SimpleBiped2a_new.SLDASM'); components={c.Name2:c for c in main.GetComponents(False)};mates=json.loads((p/'mates.json').read_text(encoding='utf8'));groups=defaultdict(list)
for m in mates:
 if m['type']!='MateConcentric' or len(m['entities'])!=2:continue
 group=m['assembly'];parent=components[group];sub=parent.GetModelDoc2;subcomp={c.Name2:c for c in sub.GetComponents(False)}; parentT=np.array(parent.Transform2.ArrayData);Rparent=parentT[:9].reshape(3,3); calculated=[]
 for e in m['entities']:
  c0=subcomp[e['component']];c1=components[group+'/'+e['component']];a=np.array(c0.Transform2.ArrayData);b=np.array(c1.Transform2.ArrayData);r0=a[:9].reshape(3,3);r1=b[:9].reshape(3,3);pt=np.array(e['parameters'][:3]);u=np.array(e['parameters'][3:6]);world=(pt-a[9:12])@r0.T@r1+b[9:12];direction=u@r0.T@r1;local=(world-parentT[9:12])@Rparent.T; localu=direction@Rparent.T;calculated.append((local,localu))
 p0,u=calculated[0];p1,v=calculated[1];error=np.linalg.norm(np.cross(p1-p0,u));
 if error>1e-6:raise RuntimeError(f'bad axis {group} {m["name"]} {error}')
 groups[group].append({'mate':m['name'],'components':[e['component'] for e in m['entities']],'axis_point_m':p0.tolist(),'axis_direction':u.tolist(),'axis_offset_m':error})
old=p/'axis_candidates_before_pose_fix.json'
if not old.exists():old.write_text((p/'axis_candidates.json').read_text(encoding='utf8'),encoding='utf8')
(p/'axis_candidates.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2),encoding='utf8');print('Updated axes:',[len(v) for v in groups.values()])
