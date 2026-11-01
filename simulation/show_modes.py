import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\single_leg_mesh_experiment.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d); j=np.asarray(d.efc_J).reshape(d.nefc,m.nv); _,s,v=np.linalg.svd(j,full_matrices=True); n=v[16:].T
for i in range(m.njnt): print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i),np.round(n[m.jnt_dofadr[i]],4))
