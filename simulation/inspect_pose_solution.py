import numpy as np,mujoco
p='C:/Practice_project/biped_sim/';m=mujoco.MjModel.from_xml_path(p+'biped_chassis_collision_candidate.xml');q=np.load(p+'balanced_initial_pose.npz')['qpos']
for i in range(m.njnt):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i)
 if m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE:
  x=q[m.jnt_qposadr[i]]
  if abs(x)>.03: print(name,round(x,4),'limited',m.jnt_limited[i],m.jnt_range[i])

