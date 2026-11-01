import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_chassis_collision_candidate.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');print('COM',d.subtree_com[body]);
for n in ['L0_visual6','L1_visual6','L0_foot_proxy','L1_foot_proxy']:
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n); print(n,d.geom_xpos[i])
print('qpos',d.qpos[:7],'contacts',d.ncon)
