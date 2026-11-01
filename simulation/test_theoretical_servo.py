import numpy as np,mujoco
from pathlib import Path
from theoretical_servo import TheoreticalServoController,ServoSpec
p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'biped_filtered_collision_experimental.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);s=TheoreticalServoController(m,d)
assert abs(ServoSpec().stall_torque_nm-2.353596)<1e-6
assert np.allclose(s.pending_us,1500)
s.set_pulse_us([20000,-20000,1500,1600,1400]);assert np.allclose(s.pending_us,[2500,500,1500,1600,1400])
# Safe small hypothetical command excursions, not hardware instructions.
s.set_pulse_us([1520,1480,1520,1480,1500]);max_constraint=0; max_floor=0; max_torque=0
for i in range(round(2/m.opt.timestep)):
 s.step();max_torque=max(max_torque,float(np.max(np.abs(d.ctrl))))
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]<16:max_constraint=max(max_constraint,abs(d.efc_pos[k]))
 for c in d.contact[:d.ncon]:
  names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  if 'ground' in names:max_floor=max(max_floor,-c.dist)
 if not np.isfinite(d.qpos).all():raise RuntimeError('Non-finite state')
print('seconds',round(d.time,3),'max torque Nm',round(max_torque,4),'joint error mm',round(max_constraint*1000,4),'ground penetration mm',round(max_floor*1000,4),'base height',round(d.qpos[2],4),'quaternion',np.round(d.qpos[3:7],4))


