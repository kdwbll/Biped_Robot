"""Update changed servo visual mesh and 55 g inertial contribution in a copy."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

p=Path(__file__).parent
before=json.loads((p/'before_servo_shaft_fix_cad_mass_inventory_current.json').read_text(encoding='utf8'))
after=json.loads((p/'cad_mass_inventory_current.json').read_text(encoding='utf8'))
def servo(rows):
 matches=[r for r in rows if r['body']=='L0_part0' and r['kind']=='servo_candidate']
 if len(matches)!=1:raise RuntimeError('Expected one L0 upper servo')
 return matches[0]
old,new=servo(before),servo(after)
if old['mass_kg']!=.055 or new['mass_kg']!=.055:raise RuntimeError('Servo mass mismatch')
tree=ET.parse(p/'diagnostic_refined_tower.xml');root=tree.getroot()
mesh=root.find("./asset/mesh[@name='L0_mesh0']")
old_file=Path(mesh.get('file'));new_file=p/'meshes_after_shaft_fix'/old_file.name
if not new_file.exists():raise RuntimeError(f'Updated STL missing: {new_file}')
mesh.set('file',new_file.relative_to(p).as_posix())
it=root.find(".//body[@name='L0_part0']/inertial");M=float(it.get('mass'));C=np.fromstring(it.get('pos'),sep=' ')
v=np.fromstring(it.get('fullinertia'),sep=' ');I=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
m=old['mass_kg'];c0=np.array(old['center_m']);c1=np.array(new['center_m'])
i0=np.array(old['approx_inertia_at_center']);i1=np.array(new['approx_inertia_at_center'])
def shift(mass,r):return mass*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
mr=M-m;cr=(M*C-m*c0)/mr
Ir=I-i0-shift(m,c0-C)-shift(mr,cr-C)
Cn=(mr*cr+m*c1)/M;In=Ir+shift(mr,cr-Cn)+i1+shift(m,c1-Cn);In=(In+In.T)/2
if np.linalg.eigvalsh(In).min()<=0:raise RuntimeError('invalid inertia')
it.set('pos',' '.join(f'{x:.12g}' for x in Cn))
it.set('fullinertia',' '.join(f'{x:.12g}' for x in [In[0,0],In[1,1],In[2,2],In[0,1],In[0,2],In[1,2]]))
root.set('model','diagnostic_post_shaft_fix');out=p/'diagnostic_post_shaft_fix.xml';ET.indent(root);tree.write(out,encoding='utf8',xml_declaration=True)
before_m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_refined_tower.xml'));after_m=mujoco.MjModel.from_xml_path(str(out))
print('Updated visual mesh:',new_file.name,'servo mass',m,'total mass',mujoco.mj_getTotalmass(after_m),
      'mass difference',mujoco.mj_getTotalmass(after_m)-mujoco.mj_getTotalmass(before_m))
print('Servo geometric center shift mm',np.round((c1-c0)*1000,5))
print('Collision proxies remain unchanged and require separate verification.')
