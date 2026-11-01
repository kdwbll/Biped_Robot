from pathlib import Path
import copy,xml.etree.ElementTree as E
import mujoco,numpy as np
p=Path(__file__).parent
root=E.Element('mujoco',model='biped_fixed_base_experiment');E.SubElement(root,'option',gravity='0 0 0',timestep='0.001',iterations='100');E.SubElement(root,'statistic',center='0.04 -0.04 0.1',extent='0.45')
a=E.SubElement(root,'asset');world=E.SubElement(root,'worldbody');E.SubElement(world,'light',pos='0 -1 1',dir='0 0 -1');eq=E.SubElement(root,'equality');tend=E.SubElement(root,'tendon');act=E.SubElement(root,'actuator')
for k,fn in enumerate(['single_leg_two_drive_candidate.xml','second_leg_two_drive_candidate.xml']):
 src=E.parse(p/fn).getroot(); prefix=f'L{k}_'
 # Rewrite all element names and references except file paths.
 for e in src.iter():
  for key in ('name','body','body1','body2','joint','joint1','joint2','tendon','mesh'):
   if key in e.attrib: e.set(key,prefix+e.get(key))
 for element in src.find('asset'):a.append(copy.deepcopy(element))
 parent=E.SubElement(world,'body',name=prefix+'mount',pos='0 0 0' if k==0 else '0 -0.041 0',quat='1 0 0 0' if k==0 else '0 1 0 0')
 for b in src.findall('./worldbody/body'):parent.append(copy.deepcopy(b))
 for section,target in [('equality',eq),('tendon',tend),('actuator',act)]:
  for element in src.find(section): target.append(copy.deepcopy(element))
E.indent(root); path=p/'two_legs_fixed_base_experiment.xml'; E.ElementTree(root).write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_forward(m,d);print('bodies',m.nbody,'meshes',m.nmesh,'constraints',m.neq,'candidate actuators',m.nu,'initial residual',np.max(np.abs(d.efc_pos)))
for j in range(m.nu):
 mujoco.mj_resetData(m,d); maxerr=0
 for i in range(1000):
  d.ctrl[:]=0;d.ctrl[j]=.0002*np.sin(i*.006);mujoco.mj_step(m,d);maxerr=max(maxerr,float(np.max(np.abs(d.efc_pos))))
 print('drive',j,'max residual m',maxerr,'finite',np.isfinite(d.qpos).all())
