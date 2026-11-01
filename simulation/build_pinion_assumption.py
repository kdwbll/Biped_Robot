from pathlib import Path
import xml.etree.ElementTree as E
import mujoco,numpy as np
p=Path(__file__).parent;tree=E.parse(p/'biped_hip_assumption_experiment.xml');root=tree.getroot();world=root.find('worldbody')
b=E.SubElement(world,'body',name='hypothetical_motor_shaft',pos='-0.0266628795 -0.006474586 0.26632');E.SubElement(b,'joint',name='hypothetical_servo_shaft',type='hinge',axis='0 0 1',damping='0.001');E.SubElement(b,'inertial',pos='0 0 0',mass='0.001',diaginertia='0.000001 0.000001 0.000001')
E.SubElement(root.find('equality'),'joint',name='estimated_pinion_23_gear_50',joint1='L0_hip_candidate',joint2='hypothetical_servo_shaft',polycoef='0 -0.46 0 0 0',solref='0.002 1')
root.find("./actuator/motor[@name='hip_candidate']").set('joint','hypothetical_servo_shaft')
E.indent(root);out=p/'biped_gear_ratio_candidate.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m); peaks=np.zeros(m.neq)
for i in range(3000):
 d.ctrl[-1]=.0005*np.sin(i*.003);mujoco.mj_step(m,d)
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY: peaks[d.efc_id[k]]=max(peaks[d.efc_id[k]],abs(d.efc_pos[k]))
print('finite',np.isfinite(d.qpos).all(),'max link error m',max(peaks[:16]),'ratio errors rad',peaks[16:])
for name in ['L0_hip_candidate','L1_hip_candidate','hypothetical_servo_shaft']:
 j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);print(name,d.qpos[m.jnt_qposadr[j]])

