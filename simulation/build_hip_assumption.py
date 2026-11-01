from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np,mujoco
p=Path(__file__).parent; tree=E.parse(p/'two_legs_fixed_base_experiment.xml');root=tree.getroot()
for i in (0,1):
 b=root.find(f"./worldbody/body[@name='L{i}_mount']")
 E.SubElement(b,'joint',name=f'L{i}_hip_candidate',type='hinge',pos='0.042 0.032 0.2',axis='0 0 1',damping='0.01',limited='true',range='-0.2 0.2')
eq=root.find('equality'); E.SubElement(eq,'joint',name='opposite_world_rotation_assumption',joint1='L1_hip_candidate',joint2='L0_hip_candidate',polycoef='0 1 0 0 0',solref='0.005 1')
E.SubElement(root.find('actuator'),'motor',name='hip_candidate',joint='L0_hip_candidate',gear='1',ctrllimited='true',ctrlrange='-0.001 0.001')
E.indent(root);out=p/'biped_hip_assumption_experiment.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d);init=float(np.max(np.abs(d.efc_pos)));peak=0
for i in range(3000):
 d.ctrl[-1]=0.0005*np.sin(i*.003);mujoco.mj_step(m,d);peak=max(peak,float(np.max(np.abs(d.efc_pos))))
print('actuators',m.nu,'initial residual',init,'max mixed constraint residual',peak,'finite',np.isfinite(d.qpos).all())
for name in ['L0_hip_candidate','L1_hip_candidate']:
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);print(name,'q',d.qpos[m.jnt_qposadr[i]])
