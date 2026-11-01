import math,mujoco,numpy as np
from pathlib import Path
m=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'biped_free_contact_experiment.xml'));m.actuator_ctrlrange[:]=[-0.5,0.5]
def addr(n):
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);return m.jnt_qposadr[i],m.jnt_dofadr[i]
up=[addr('L0_part1_r'),addr('L1_part1_r')];knee=[(addr('L0_part5_r'),addr('L0_part3_r')),(addr('L1_part5_r'),addr('L1_part3_r'))];turn=addr('hypothetical_servo_shaft')
for amplitude in (0,.03,.06,.1):
 d=mujoco.MjData(m);minheight=100;maxerr=0
 for i in range(5000):
  t=i*m.opt.timestep; targets=[amplitude*math.sin(t*4),amplitude*math.sin(t*4+math.pi)]
  for k in range(2):
   q,v=up[k]; a,b=knee[k]; d.ctrl[k*2]=np.clip(2*(targets[k]-d.qpos[q])-.05*d.qvel[v],-.5,.5); d.ctrl[k*2+1]=np.clip(-2*(d.qpos[a[0]]-d.qpos[b[0]])-.05*(d.qvel[a[1]]-d.qvel[b[1]]),-.5,.5)
  d.ctrl[4]=np.clip(-2*d.qpos[turn[0]]-.05*d.qvel[turn[1]],-.5,.5);mujoco.mj_step(m,d);minheight=min(minheight,d.qpos[2])
  if not np.isfinite(d.qpos).all():break
 print('amp',amplitude,'time',round(d.time,2),'xy',np.round(d.qpos[:2],4),'min_base_z',round(minheight,4),'quat',np.round(d.qpos[3:7],3))
