from pathlib import Path
import mujoco,numpy as np
p=Path(r'C:\Practice_project\biped_sim');m=mujoco.MjModel.from_xml_path(str(p/'biped_chassis_collision_candidate.xml'));d=mujoco.MjData(m);d.qpos[:]=np.load(p/'balanced_initial_pose.npz')['qpos'];mujoco.mj_forward(m,d)
print('contacts',d.ncon,'min contact distance',min((d.contact[i].dist for i in range(d.ncon)),default=0))
for name in ('L0_visual6','L1_visual6'):
 g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name);mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh];v=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];print(name,'world bbox min',v.min(axis=0),'max',v.max(axis=0))
