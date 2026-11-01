"""Inspect coincident planar facets with a detection-only contact margin."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

p=Path(__file__).parent
tree=ET.parse(p/'diagnostic_cad_plane_contact_trial.xml');root=tree.getroot();count=0
size=root.find('size')
if size is None:size=ET.SubElement(root,'size')
size.set('memory','256M')
for pair in root.find('contact').findall('pair'):
 if (pair.get('geom1') or '').startswith('plane_'):
  pair.set('margin','.000001');pair.set('gap','.000001');count+=1
root.set('model','diagnostic_planar_detection_only');path=p/'diagnostic_planar_detection_only.xml'
tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
rows=[];leg_samples={0:[],1:[]}
for i,c in enumerate(d.contact[:d.ncon]):
 a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or ''
 b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
 if a.startswith('plane_') and b.startswith('plane_'):
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  row={'dist_m':float(c.dist),'efc_address':int(c.efc_address),'normal_force_n':float(f[0])}
  leg=0 if 'plane_L0_' in a or 'plane_L0_' in b else 1
  if len(leg_samples[leg])<100:
   ja=np.zeros((3,m.nv));jb=np.zeros((3,m.nv))
   mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[c.geom1]))
   mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[c.geom2]))
   row['normal_relative_jacobian_norm']=float(np.linalg.norm(np.asarray(c.frame[:3])@(ja-jb)))
   leg_samples[leg].append(row['normal_relative_jacobian_norm'])
  rows.append(row)
warnings={name:int(d.warning[int(getattr(mujoco.mjtWarning,name))].number) for name in ('mjWARN_CONTACTFULL','mjWARN_CNSTRFULL')}
total_force=sum(x['normal_force_n'] for x in rows)
out={'valid_for_dynamics':not any(warnings.values()) and np.isfinite(total_force) and total_force<500,
 'warnings':warnings,'pairs':count,'total_contacts':d.ncon,'plane_contact_count':len(rows),
 'plane_active_constraint_count':sum(x['efc_address']>=0 for x in rows),
 'plane_normal_force_sum_n':total_force,
 'min_plane_distance_mm':min((x['dist_m']*1000 for x in rows),default=None),
 'max_plane_distance_mm':max((x['dist_m']*1000 for x in rows),default=None),
 'sampled_normal_relative_jacobian_norm_max_by_leg':{str(k):max(v,default=None) for k,v in leg_samples.items()},
 'sampled_jacobian_counts_by_leg':{str(k):len(v) for k,v in leg_samples.items()},
 'limitations':'Single initial-pose experiment. Equal margin/gap does not prevent numerical overconstraint here; total force is an invalid numerical result. 500 N threshold is an arbitrary diagnostic stop, not a physical limit.'}
(p/'planar_detection_only_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
