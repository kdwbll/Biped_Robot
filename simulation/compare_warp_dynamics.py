"""Compare an uncontrolled model for up to 1 second; not an RL parity test."""
import json,time
from pathlib import Path
import mujoco,numpy as np,warp as wp,mujoco_warp as mjw
root=Path(__file__).parent;wp.config.kernel_cache_dir=str(root/'warp_cache');wp.init()
m=mujoco.MjModel.from_xml_path(str(root/'biped_filtered_collision_experimental.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out=[]
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=1,nconmax=1024,njmax=4096)
 with wp.ScopedCapture() as c:mjw.step(gm,gd)
 for k in range(2000):
  wp.capture_launch(c.graph);mujoco.mj_step(m,d)
  if k+1 in (20,200,1000,2000):
   wp.synchronize();q=gd.qpos.numpy()[0];v=gd.qvel.numpy()[0]
   row=dict(seconds=(k+1)*m.opt.timestep,max_qpos_error=float(np.max(np.abs(q-d.qpos))),
    max_qvel_error=float(np.max(np.abs(v-d.qvel))),finite=bool(np.isfinite(q).all() and np.isfinite(v).all()))
   out.append(row);print(json.dumps(row),flush=True)
(root/'warp_dynamics_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8')
