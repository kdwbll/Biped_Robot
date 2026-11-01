import mujoco,numpy as np
p=r'C:\Practice_project\biped_sim\single_leg_mesh_experiment.xml'; m=mujoco.MjModel.from_xml_path(p); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
J=np.asarray(d.efc_J).reshape(d.nefc,m.nv); u,s,v=np.linalg.svd(J,full_matrices=True); null=v[16:].T
ix=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in ['part1_r','part2_r']]; print('remaining motion modes',null.shape[1],'drive mapping rank',np.linalg.matrix_rank(null[ix,:]),'determinant',np.linalg.det(null[ix,:]))
