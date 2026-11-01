import numpy as np,mujoco
p=r'C:\Practice_project\biped_sim\biped_hip_assumption_experiment.xml';m=mujoco.MjModel.from_xml_path(p);d=mujoco.MjData(m);peaks=np.zeros(m.neq)
for i in range(3000):
 d.ctrl[:]=0;d.ctrl[4]=.0005*np.sin(i*.003);mujoco.mj_step(m,d)
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY:
   peaks[d.efc_id[k]]=max(peaks[d.efc_id[k]],abs(d.efc_pos[k]))
print('max pin error m',max(peaks[:16]));print('gear angle mismatch rad',peaks[16]);print('finite',np.isfinite(d.qpos).all())
