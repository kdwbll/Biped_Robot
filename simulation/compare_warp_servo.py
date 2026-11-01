"""Compare CPU and GPU servo trajectories; no support forces or RL."""
import json,time
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from stabilized_servo import StabilizedServo
from warp_servo import WarpServo,STALL
root=Path(__file__).parent;wp.config.kernel_cache_dir=str(root/'warp_cache');wp.init()
m=mujoco.MjModel.from_xml_path(str(root/'biped_filtered_collision_experimental.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
cpu=StabilizedServo(m,d)
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=1,nconmax=1024,njmax=4096)
 servo=WarpServo(gm,gd,cpu.initial_positions,1);rows=[]
 for k in range(100):
  phase=2*np.pi*k*.02;a=np.array([-.5*max(0,np.sin(phase)),.5*max(0,np.sin(phase)),.5*max(0,-np.sin(phase)),-.5*max(0,-np.sin(phase)),0.])
  servo.set_actions(a[None,:]);servo.control_step();cpu.set_pulse_us(1500.+180.*a)
  for _ in range(40):cpu.step()
  if k+1 in (1,5,25,50,100):
   wp.synchronize();q=gd.qpos.numpy()[0];v=gd.qvel.numpy()[0]
   row=dict(seconds=(k+1)*.02,max_qpos_error=float(np.max(np.abs(q-d.qpos))),
    max_qvel_error=float(np.max(np.abs(v-d.qvel))),finite=bool(np.isfinite(q).all() and np.isfinite(v).all()))
   rows.append(row);print(json.dumps(row),flush=True)
 (root/'warp_servo_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
