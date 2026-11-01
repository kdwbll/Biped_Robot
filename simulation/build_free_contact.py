from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np,mujoco
p=Path(__file__).parent;t=E.parse(p/'biped_gear_ratio_candidate.xml');root=t.getroot();opt=root.find('option');opt.set('gravity','0 0 -9.81');opt.set('timestep','0.001');world=root.find('worldbody')
robot=E.SubElement(world,'body',name='robot_free',pos='0 0 0.086');E.SubElement(robot,'freejoint',name='floating_base');E.SubElement(robot,'inertial',pos='0.04 -0.02 0.2',mass='0.7',diaginertia='0.004 0.004 0.004')
E.SubElement(robot,'geom',name='chassis_collision_proxy',type='box',pos='0.041 -0.0205 0.206',size='0.096 0.0875 0.07',mass='0',friction='0.8 0.01 0.001',rgba='0.5 0.5 0.5 0.2')
for name in ['L0_mount','L1_mount','hypothetical_motor_shaft']:
 b=world.find(f"body[@name='{name}']");world.remove(b);robot.append(b)
g=world.find("geom[@name='chassis_visual']");world.remove(g);robot.append(g)
E.SubElement(world,'geom',name='ground',type='plane',size='2 2 0.1',friction='0.8 0.01 0.001',rgba='0.35 0.4 0.45 1')
foot0=root.find(".//body[@name='L0_part6']");foot1=root.find(".//body[@name='L1_part6']")
E.SubElement(foot0,'geom',name='L0_foot_proxy',type='box',pos='0.07476 0.0408 -0.08103',size='0.0627 0.0339 0.004',mass='0',friction='0.8 0.01 0.001',rgba='0.2 0.6 0.3 0.3')
E.SubElement(foot1,'geom',name='L1_foot_proxy',type='box',pos='0.04153 0.045 0.07436',size='0.0625 0.0375 0.004',mass='0',friction='0.8 0.01 0.001',rgba='0.2 0.6 0.3 0.3')
E.indent(root);out=p/'biped_free_contact_experiment.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d);print('initial contacts',d.ncon,'mass',round(mujoco.mj_getTotalmass(m),3),'nu',m.nu,'initial root z',d.qpos[2]);maxp=0
for i in range(1000):
 mujoco.mj_step(m,d)
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]<16:maxp=max(maxp,abs(d.efc_pos[k]))
 if not np.isfinite(d.qpos).all():break
print('end time',d.time,'base z',d.qpos[2],'contacts',d.ncon,'max link error',maxp,'finite',np.isfinite(d.qpos).all())

