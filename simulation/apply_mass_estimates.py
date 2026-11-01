from pathlib import Path
import json,math,xml.etree.ElementTree as E
from collections import defaultdict
import numpy as np,mujoco
p=Path(r'C:\Practice_project\biped_sim');data=json.loads((p/'cad_mass_inventory.json').read_text(encoding='utf8'));items=defaultdict(list)
for row in data:items[row['body']].append((row['mass_kg'],np.array(row['center_m']),np.array(row['approx_inertia_at_center'])))
t=E.parse(p/'self_collision_split_candidate.xml');root=t.getroot();shaftmass=0
for body in root.iter('body'):
 name=body.get('name')
 for g in body.findall('geom'):
  if not (g.get('name') or '').startswith('visual_pin_'):continue
  xyz=np.array([float(x) for x in g.get('fromto').split()]);a,b=xyz[:3],xyz[3:];center=(a+b)/2;L=np.linalg.norm(b-a);u=(b-a)/L;r=float(g.get('size'));mass=8000*math.pi*r*r*L;I=mass*(3*r*r+L*L)/12*(np.eye(3)-np.outer(u,u))+mass*r*r/2*np.outer(u,u);items[name].append((mass,center,I));shaftmass+=mass
for name,entries in items.items():
 body=root.find(f".//body[@name='{name}']")
 if body is None:raise RuntimeError(name)
 M=sum(x[0] for x in entries);center=sum(m*c for m,c,I in entries)/M;inertia=np.zeros((3,3))
 for mass,c,I in entries:
  r=c-center;inertia+=I+mass*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
 inertia=(inertia+inertia.T)/2; ev=np.linalg.eigvalsh(inertia)
 if ev.min()<=0:raise RuntimeError((name,ev))
 old=body.find('inertial')
 if old is not None:body.remove(old)
 E.SubElement(body,'inertial',pos=' '.join(f'{x:.12g}' for x in center),mass=f'{M:.12g}',fullinertia=' '.join(f'{x:.12g}' for x in [inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))

for eq in root.findall('./equality/connect'): eq.set('solref','0.001 1'); eq.set('solimp','0.99 0.999 0.001')
root.set('model','biped_estimated_mass_collision');E.indent(root);out=p/'biped_estimated_mass_collision.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
print('CAD mass kg',sum(r['mass_kg'] for r in data),'assumed pin mass kg',shaftmass,'model total kg',mujoco.mj_getTotalmass(m),'contacts',d.ncon)
for i in range(4000):mujoco.mj_step(m,d)
print('time',d.time,'qpos finite',np.isfinite(d.qpos).all(),'final contacts',d.ncon)

