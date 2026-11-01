import mujoco,numpy as np
from pathlib import Path
from biped_env import BipedEnv
p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'biped_filtered_collision_experimental.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
for step in range(8000):
 mujoco.mj_step(m,d)
 if step in (999,1999,3999,7999):
  R=d.xmat[idx].reshape(3,3);print('raw viewer t',d.time,'roll degrees',round(float(np.degrees(np.arctan2(R[2,1],R[2,2]))),2),'up',round(float(R[2,2]),3),'height',round(float(d.xipos[idx,2]),3))
