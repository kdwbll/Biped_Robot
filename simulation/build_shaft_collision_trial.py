from pathlib import Path
import xml.etree.ElementTree as E
import json,numpy as np,mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_upper_collision_trial.xml');root=t.getroot();contact=root.find('contact');sets=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).values());geoms=[g.get('name','') for g in root.iter('geom')]
for leg,axes in enumerate(sets):
 parts=sorted({v for a in axes for v in a['components']});unique=[]
 for a in axes:
  q=np.array(a['axis_point_m']);u=np.array(a['axis_direction']);u/=np.linalg.norm(u);match=next((x for x in unique if np.linalg.norm(np.cross(q-x['q'],x['u']))<1e-5),None)
  if match:match['parts'].update(parts.index(c) for c in a['components'])
  else:unique.append({'q':q,'u':u,'parts':set(parts.index(c) for c in a['components'])})
 for i,x in enumerate(unique):
  pin=f'visual_pin_{leg}_{i}';E.SubElement(contact,'pair',geom1='ground',geom2=pin,solref='.001 1',solimp='.99 .999 .001')
  for name in geoms:
   isother=name.startswith((f'col_L{1-leg}_',f'knee_L{1-leg}_',f'upper_L{1-leg}_'))
   isown=any(name.startswith(f'col_L{leg}_part{n}_') for n in range(1,7) if n not in x['parts']) or (3 not in x['parts'] and name.startswith(f'knee_L{leg}_part3_')) or (0 not in x['parts'] and name.startswith(f'upper_L{leg}_part0_'))
   if isother or isown or name.startswith('hull_chassis_'):E.SubElement(contact,'pair',geom1=pin,geom2=name,solref='.001 1',solimp='.99 .999 .001')
pins=[n for n in geoms if n.startswith('visual_pin_')]
for i,a in enumerate(pins):
 for b in pins[i+1:]: E.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','biped_shaft_collision_trial');E.indent(root);out=p/'biped_shaft_collision_trial.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);bad=[]
for i in range(d.ncon):
 c=d.contact[i]
 if c.dist<-.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('initial contacts',d.ncon,'initial overlap >1mm',len(bad),bad[:8])


