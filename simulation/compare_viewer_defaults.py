from pathlib import Path
import numpy as np,mujoco
base=Path(__file__).parent
for name in ['biped_with_visual_pins.xml','biped_chassis_collision_candidate.xml','biped_filtered_collision_experimental.xml']:
 for key in (False,True):
  m=mujoco.MjModel.from_xml_path(str(base/name));d=mujoco.MjData(m)
  if key and not m.nkey:continue
  if key:mujoco.mj_resetDataKeyframe(m,d,0)
  mujoco.mj_forward(m,d);i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');peak=0
  for k in range(round(2/m.opt.timestep)):
   mujoco.mj_step(m,d);R=d.xmat[i].reshape(3,3);roll=float(np.degrees(np.arctan2(R[2,1],R[2,2])));peak=max(peak,abs(roll))
  print(name,'key',key,'peak roll',round(peak,2),'final',round(roll,2),'height',round(d.xipos[i,2],3),'finite',np.isfinite(d.qpos).all())
