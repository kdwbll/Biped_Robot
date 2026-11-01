from pathlib import Path
import json,xml.etree.ElementTree as E
import mujoco,numpy as np
p=Path(__file__).parent; sets=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).values());t=E.parse(p/'biped_free_contact_experiment.xml');root=t.getroot()
for leg,axes in enumerate(sets):
 parts=sorted({v for a in axes for v in a['components']});unique=[]
 for a in axes:
  q=np.array(a['axis_point_m']);u=np.array(a['axis_direction']);u=u/np.linalg.norm(u)
  if any(np.linalg.norm(np.cross(q-x['q'],x['u']))<1e-5 for x in unique):continue
  unique.append({'q':q,'u':u,'component':a['components'][0]})
 for i,item in enumerate(unique):
  q=item['q'];u=item['u'];
  if abs(u[1])<.9: raise ValueError('Unexpected axle direction')
  center=q+u*((.032-q[1])/u[1]); a=center-u*.045;b=center+u*.045
  body=root.find(f".//body[@name='L{leg}_part{parts.index(item['component'])}']")
  E.SubElement(body,'geom',name=f'visual_pin_{leg}_{i}',type='cylinder',**{'fromto':' '.join(f'{v:.9g}' for v in [*a,*b]),'size':'0.002','contype':'0','conaffinity':'0','mass':'0','rgba':'0.9 0.55 0.05 1'})
 print('leg',leg,'unique shaft axes',len(unique))
for geom in root.iter('geom'):
 if geom.get('type') == 'capsule': geom.set('rgba','0 0 0 0')
E.indent(root);out=p/'biped_with_visual_pins.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
for _ in range(100):mujoco.mj_step(m,d)
print('loaded meshes',m.nmesh,'equalities',m.neq,'finite',np.isfinite(d.qpos).all())

