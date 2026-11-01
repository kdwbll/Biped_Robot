"""Batched servo + policy inference benchmark; NOT a PPO training benchmark."""
import json,math,sys,time
from pathlib import Path
import numpy as np,mujoco,torch,warp as wp,mujoco_warp as mjw
from stabilized_servo import StabilizedServo
from warp_servo import WarpServo
root=Path(__file__).parent;wp.config.kernel_cache_dir=str(root/'warp_cache');wp.init()
torch.set_num_threads(1);torch.manual_seed(42)
n=int(sys.argv[1]) if len(sys.argv)>1 else 32;steps=200
m=mujoco.MjModel.from_xml_path(str(root/'biped_filtered_collision_experimental.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);cpu=StabilizedServo(m,d)
body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
policy=torch.nn.Sequential(torch.nn.Linear(18,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,5),torch.nn.Tanh()).cuda().eval()
with wp.ScopedDevice('cuda:0'),torch.no_grad():
 gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=n,nconmax=1024,njmax=4096);servo=WarpServo(gm,gd,cpu.initial_positions,n)
 previous=torch.zeros((n,5),device='cuda');initial=wp.to_torch(servo.initial);target=wp.to_torch(servo.targets)
 xmat=wp.to_torch(gd.xmat);ximat=wp.to_torch(gd.ximat);cvel=wp.to_torch(gd.cvel);xipos=wp.to_torch(gd.xipos)
 start=time.perf_counter();forward=torch.zeros(n,device='cuda');before=xipos[:,body,0].clone()
 for k in range(steps):
  # Conservative synchronization makes interop correct but costs throughput.
  wp.synchronize();rot=xmat[:,body];gravity=-rot[:,2,:];omega=torch.bmm(ximat[:,body].transpose(1,2),cvel[:,body,:3].unsqueeze(-1)).squeeze(-1)
  phase=2*math.pi*k*.02;clock=torch.tensor([math.sin(phase),math.cos(phase)],device='cuda').expand(n,-1)
  obs=torch.cat((gravity,omega,previous,target-initial,clock),dim=1)
  residual=policy(obs);left=max(0.,math.sin(phase));right=max(0.,-math.sin(phase))
  reference=torch.tensor([-left,left,right,-right,0.],device='cuda').expand(n,-1)
  actions=(reference+.3*residual).clamp(-1,1).contiguous();torch.cuda.synchronize()
  servo.apply_support(body,1.3125,.35*sum(m.body_mass)*9.81);servo.set_actions_device(actions);servo.control_step();previous=actions
 wp.synchronize();torch.cuda.synchronize();elapsed=time.perf_counter()-start
 overflow=gd.overflow.numpy();q=gd.qpos.numpy()
 out=dict(worlds=n,control_steps=steps*n,seconds=elapsed,control_steps_per_sec=steps*n/elapsed,
  physics_per_control=40,finite=bool(np.isfinite(q).all()),overflow=overflow.tolist(),
  scope='random policy inference and servo/physics; no episode reset, contact rewards, PPO updates or rendering')
 print('RESULT '+json.dumps(out),flush=True)
