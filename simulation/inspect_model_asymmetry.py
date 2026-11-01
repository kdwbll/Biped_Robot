import numpy as np,mujoco
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim');m=mujoco.MjModel.from_xml_path(str(p/'biped_filtered_collision_experimental.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
for body in ['L0_part0','L1_part0']:
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body);print(body,'mass kg',m.body_mass[i],'ipos',m.body_ipos[i], 'principal inertia',m.body_inertia[i])
for name in ['L0_upper_candidate','L0_knee_candidate','L1_upper_candidate','L1_knee_candidate','hip_candidate']:
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,name); print(name,'ctrlrange',m.actuator_ctrlrange[i],'gear',m.actuator_gear[i,0])
