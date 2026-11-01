import mujoco,numpy as np
p=r'C:\Practice_project\biped_sim\single_leg_mesh_experiment.xml'; m=mujoco.MjModel.from_xml_path(p); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
J=np.asarray(d.efc_J).reshape(d.nefc,m.nv); s=np.linalg.svd(J,compute_uv=False); print('dofs',m.nv,'rank',np.linalg.matrix_rank(J,tol=1e-8),'remaining',m.nv-np.linalg.matrix_rank(J,tol=1e-8)); print('small singular values',s[-6:])
