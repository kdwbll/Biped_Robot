from pathlib import Path
import xml.etree.ElementTree as E
import os,mujoco,numpy as np
p=Path(__file__).parent; leg=int(os.environ.get('BIPED_LEG_INDEX','0')); tree=E.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')); root=tree.getroot()
t=E.SubElement(root,'tendon'); f=E.SubElement(t,'fixed',name='knee_relative_angle'); E.SubElement(f,'joint',joint='part5_r',coef='1'); E.SubElement(f,'joint',joint='part3_r',coef='-1')
a=E.SubElement(root,'actuator'); E.SubElement(a,'motor',name='upper_candidate',joint='part1_r',gear='1',ctrllimited='true',ctrlrange='-0.001 0.001'); E.SubElement(a,'motor',name='knee_candidate',tendon='knee_relative_angle',gear='1',ctrllimited='true',ctrlrange='-0.001 0.001')
E.indent(root); path=p/('single_leg_two_drive_candidate.xml' if leg==0 else 'second_leg_two_drive_candidate.xml'); tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path)); d=mujoco.MjData(m)
for actuator in range(2):
 mujoco.mj_resetData(m,d); peakerr=0;peakmove=0
 for i in range(2000):
  d.ctrl[:]=0; d.ctrl[actuator]=0.0002*np.sin(i*.004);mujoco.mj_step(m,d)
  peakerr=max(peakerr,float(np.max(np.abs(d.efc_pos)))); peakmove=max(peakmove,float(np.max(np.abs(d.qpos))))
 print('candidate',actuator,'max constraint error m',peakerr,'movement',peakmove,'finite',np.isfinite(d.qpos).all())


