import mujoco,numpy as np
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim')
for f in ['single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml']:
 m=mujoco.MjModel.from_xml_path(str(p/f));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 J=np.asarray(d.efc_J).reshape(d.nefc,m.nv); rank=np.linalg.matrix_rank(J,tol=1e-8); ix=m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'part1_r')]
 maxerr=0; displacement=0
 for t in range(2000):
  d.qfrc_applied[ix]=0.0002*np.sin(t*.004);mujoco.mj_step(m,d)
  maxerr=max(maxerr,float(np.max(np.abs(d.efc_pos))));displacement=max(displacement,float(np.max(np.abs(d.qpos))))
 print(f,'constraint rank',rank,'free coordinates',m.nv-rank,'max residual',maxerr,'movement',displacement,'finite',np.isfinite(d.qpos).all())
