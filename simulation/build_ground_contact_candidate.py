from pathlib import Path
import xml.etree.ElementTree as E
import mujoco,numpy as np
p=Path(__file__).parent;t=E.parse(p/'biped_with_visual_pins.xml');root=t.getroot();root.find('option').set('timestep','0.0005'); root.find('option').set('integrator','implicitfast'); contact=E.SubElement(root,'contact')
for i in (0,1):
 for j in range(7):
  visual=root.find(f".//geom[@name='L{i}_visual{j}']"); visual.set('group','1')
  E.SubElement(contact,'pair',name=f'floor_L{i}_{j}',geom1='ground',geom2=f'L{i}_visual{j}',friction='0.8 0.01 0.001 0.0001 0.0001',solref='0.001 1',solimp='0.99 0.999 0.001')
 for g in root.iter('geom'):
  if g.get('name')==f'L{i}_foot_proxy':g.set('contype','0');g.set('conaffinity','0');g.set('rgba','0 0 0 0')
E.SubElement(contact,'pair',name='feet_candidate',geom1='L0_visual6',geom2='L1_visual6',friction='0.8 0.01 0.001 0.0001 0.0001',solref='0.001 1',solimp='0.99 0.999 0.001')
E.indent(root);out=p/'biped_ground_contact_candidate.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
print('initial contact',d.ncon,'min distance', min((d.contact[i].dist for i in range(d.ncon)), default=0))
peak=0;minground=0
for i in range(4000):
 mujoco.mj_step(m,d)
 for c in d.contact[:d.ncon]:
  ids=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  if 'ground' in ids:minground=min(minground,c.dist)
 if not np.isfinite(d.qpos).all():break
print('time',d.time,'lowest ground contact distance m',minground,'finite',np.isfinite(d.qpos).all(),'final contacts',d.ncon)

