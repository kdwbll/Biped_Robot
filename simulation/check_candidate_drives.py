import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\single_leg_mesh_experiment.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d); J=np.asarray(d.efc_J).reshape(d.nefc,m.nv); _,s,V=np.linalg.svd(J,full_matrices=True); N=V[np.linalg.matrix_rank(J,tol=1e-8):].T
ids={f'part{i}':m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'part{i}_r')] for i in range(1,7)}
M=np.vstack((N[ids['part1']], N[ids['part4']]-N[ids['part1']])); print('hypothetical independent coordinates rank',np.linalg.matrix_rank(M),'matrix',M)
