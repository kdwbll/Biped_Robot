import os,mujoco,numpy as np
p=os.path.join(r'C:\Practice_project\biped_sim', 'second_leg_constraint_experiment.xml' if os.environ.get('BIPED_LEG_INDEX')=='1' else 'single_leg_constraint_experiment.xml')
m=mujoco.MjModel.from_xml_path(p); d=mujoco.MjData(m)
idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'part1_r'); adr=m.jnt_dofadr[idx]
maxerr=0; maxmove=0
for i in range(2000):
 d.qfrc_applied[adr]=0.0002*np.sin(i*0.004)
 mujoco.mj_step(m,d)
 if d.nefc: maxerr=max(maxerr,float(np.max(np.abs(d.efc_pos))))
 maxmove=max(maxmove,float(np.max(np.abs(d.qpos))))
print('driven 2s max residual m',maxerr,'max generalized movement',maxmove,'finite',np.isfinite(d.qpos).all())

