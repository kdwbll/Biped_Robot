import mujoco,numpy as np
p=r'C:\Practice_project\biped_sim\biped_free_contact_experiment.xml';m=mujoco.MjModel.from_xml_path(p);d=mujoco.MjData(m);m.actuator_ctrlrange[:]=[-0.5,0.5]
def jaddr(n):
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);return m.jnt_qposadr[i],m.jnt_dofadr[i]
hips=[jaddr('L0_part1_r'),jaddr('L1_part1_r')]; knees=[(jaddr('L0_part5_r'),jaddr('L0_part3_r')),(jaddr('L1_part5_r'),jaddr('L1_part3_r'))];turn=jaddr('hypothetical_servo_shaft');peak=0
for i in range(10000):
 vals=[]
 for k in range(2):
  q,v=hips[k]; vals.append((-2*d.qpos[q]-.05*d.qvel[v])); (a,b)=knees[k]; vals.append(-2*(d.qpos[a[0]]-d.qpos[b[0]])-.05*(d.qvel[a[1]]-d.qvel[b[1]]))
 d.ctrl[:4]=np.clip(vals,-.5,.5);d.ctrl[4]=np.clip(-2*d.qpos[turn[0]]-.05*d.qvel[turn[1]],-.5,.5);mujoco.mj_step(m,d)
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]<16:peak=max(peak,abs(d.efc_pos[k]))
 if not np.isfinite(d.qpos).all(): break
print('time',d.time,'base z',d.qpos[2],'quat',d.qpos[3:7],'contacts',d.ncon,'max position constraint m',peak,'finite',np.isfinite(d.qpos).all())

