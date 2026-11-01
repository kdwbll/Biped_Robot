import json, os, xml.etree.ElementTree as E
from pathlib import Path
import mujoco,numpy as np
p=Path(__file__).parent
sets=json.loads((p/'axis_candidates.json').read_text(encoding='utf-8')); leg_index=int(os.environ.get('BIPED_LEG_INDEX','0')); axes=list(sets.values())[leg_index]
parts=sorted({x for a in axes for x in a['components']}); rootpart=axes[0]['components'][0]; names={part:'part'+str(i) for i,part in enumerate(parts)}
root=E.Element('mujoco',model='single_leg_constraint_experiment')
E.SubElement(root,'option',gravity='0 0 0',timestep='0.001',iterations='100')
world=E.SubElement(root,'worldbody'); E.SubElement(world,'light',pos='0 -1 1'); E.SubElement(world,'camera',name='side',pos='0.04 -0.7 0.04',xyaxes='1 0 0 0 0 1')
for part in parts:
 b=E.SubElement(world,'body',name=names[part])
 if part!=rootpart:
  E.SubElement(b,'joint',name=names[part]+'_x',type='slide',axis='1 0 0',damping='0.1')
  E.SubElement(b,'joint',name=names[part]+'_z',type='slide',axis='0 0 1',damping='0.1')
  E.SubElement(b,'joint',name=names[part]+'_r',type='hinge',axis='0 1 0',damping='0.002')
 points=[a['axis_point_m'] for a in axes if part in a['components']]
 points=[[x[0],0,x[2]] for x in points]
 uniq=[]
 for pt in points:
  if all(np.linalg.norm(np.array(pt)-np.array(x))>0.0001 for x in uniq): uniq.append(pt)
 if len(uniq)>1:
  pair=max(((x,y) for i,x in enumerate(uniq) for y in uniq[i+1:]),key=lambda z:np.linalg.norm(np.array(z[0])-z[1]))
  E.SubElement(b,'geom',type='capsule',**{'fromto':' '.join(map(str,pair[0]+pair[1])),'size':'0.003','mass':'0.05','contype':'0','conaffinity':'0'})
 else:
  E.SubElement(b,'geom',type='sphere',pos=' '.join(map(str,uniq[0])),size='0.004',mass='0.05',contype='0',conaffinity='0')
eq=E.SubElement(root,'equality')
for i,a in enumerate(axes):
 x,y,z=a['axis_point_m']; E.SubElement(eq,'connect',name='pin'+str(i),body1=names[a['components'][0]],body2=names[a['components'][1]],anchor=f'{x} 0 {z}',solref='0.005 1')
E.indent(root); path=p/('single_leg_constraint_experiment.xml' if leg_index == 0 else 'second_leg_constraint_experiment.xml'); E.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path)); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
print('bodies',m.nbody-1,'constraints',m.neq,'dofs',m.nv,'initial max residual',float(np.max(np.abs(d.efc_pos))))
for i in range(1000): mujoco.mj_step(m,d)
print('1s zero gravity: max residual',float(np.max(np.abs(d.efc_pos))),'finite',bool(np.all(np.isfinite(d.qpos))))

