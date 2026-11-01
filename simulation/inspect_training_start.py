from pathlib import Path
import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_shaft_collision_trial.xml');d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');print('chassis COM',d.xipos[i],'body origin',d.xpos[i],'up',d.xmat[i].reshape(3,3)[2,2])
