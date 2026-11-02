"""Build an independent ideal lubricated-hip diagnostic candidate."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent;src=p/'diagnostic_chassis0_collision_trial.xml'
tree=ET.parse(src);root=tree.getroot();changes=[]
for leg in (0,1):
 name=f'L{leg}_hip_candidate';joint=root.find(f".//joint[@name='{name}']")
 if joint is None:raise RuntimeError('Hip joint missing')
 before=dict(joint.attrib);joint.set('damping','0');joint.set('frictionloss','0')
 changes.append({'joint':name,'before':before,'after':dict(joint.attrib)})
root.set('model','diagnostic_smooth_hip_trial');path=p/'diagnostic_smooth_hip_trial.xml'
tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'model':path.name,'changes':changes,'nq':m.nq,'ngeom':m.ngeom,'contact_pairs':m.npair,
 'limitations':'Zero hip damping and friction are ideal lubrication assumptions, not measured hardware; hinge removes the CAD radial clearance. Other exclusions remain; not for training.'}
(p/'smooth_hip_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
