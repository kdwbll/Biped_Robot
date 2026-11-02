"""Guarded four-world GPU stepping proof-of-concept; never trains."""
from pathlib import Path
from dataclasses import replace
import json,sys,time
import mujoco,numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
import mujoco_warp as mjw
from sector_warp_guard import scan_state
from sector_warp_servo_kernels import servo_step
wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));m.opt.timestep=.00025
cpu=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,cpu,0);mujoco.mj_forward(m,cpu)
s=CandidateStandingServo(m,cpu,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
h=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)];shaft=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')])
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,cpu,nworld=4,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize()
 q=gd.qpos.numpy();v=gd.qvel.numpy();q[1,h[0]]+=np.deg2rad(4.6);q[2,h[0]]+=np.deg2rad(.3);v[3,0]=np.nan
 wp.copy(gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'));wp.copy(gd.qvel,wp.array(v,dtype=wp.float32,device='cuda:0'))
 flags=wp.zeros(4,dtype=wp.int32,device='cuda:0');args=[gd.qpos,gd.qvel,flags,m.nq,m.nv,h[0],h[1],shaft,float(cpu.qpos[h[0]]),float(cpu.qpos[h[1]]),float(cpu.qpos[shaft]),float(np.deg2rad(4.5)),float(np.deg2rad(.2))]
 before=gd.time.numpy().copy();wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();injected=flags.numpy().copy();after=gd.time.numpy().copy()
 if injected.tolist()!=[0,6,4,1] or not np.array_equal(before,after):raise RuntimeError(('injected state not blocked',injected.tolist()))
 # Create fresh data; the invalid batch is discarded rather than stepped.
 gd=mjw.put_data(m,cpu,nworld=4,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize();flags=wp.zeros(4,dtype=wp.int32,device='cuda:0')
 args=[gd.qpos,gd.qvel,flags,m.nq,m.nv,h[0],h[1],shaft,float(cpu.qpos[h[0]]),float(cpu.qpos[h[1]]),float(cpu.qpos[shaft]),float(np.deg2rad(4.5)),float(np.deg2rad(.2))]
 targets=wp.array(np.tile(np.asarray(s.initial_positions,dtype=np.float32),(4,1)),dtype=wp.float32,device='cuda:0');integral=wp.zeros((4,5),dtype=wp.float32,device='cuda:0')
 with wp.ScopedCapture() as cap:
  wp.launch(servo_step,dim=(4,5),inputs=[targets,integral,gd.actuator_length,gd.actuator_velocity,gd.ctrl,m.opt.timestep,0.]);mjw.step(gm,gd)
 graph=cap.graph;start=time.perf_counter();previous=gd.time.numpy().copy();stop=None
 for step in range(80):
  wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();bad=flags.numpy();overflow=gd.overflow.numpy();now=gd.time.numpy()
  if np.any(bad) or np.any(overflow):stop='state_guard_or_overflow';break
  if step>0 and np.any(now<previous+.5*m.opt.timestep):stop='time_reset';break
  previous=now.copy();wp.capture_launch(graph)
 wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();bad=flags.numpy();overflow=gd.overflow.numpy();finaltime=gd.time.numpy()
 if np.any(bad) or np.any(overflow):stop='state_guard_or_overflow'
 if np.any(finaltime<previous+.5*m.opt.timestep):stop='time_reset'
 elapsed=time.perf_counter()-start
out={'worlds':4,'injected_flags_before_any_step':injected.tolist(),'invalid_batch_physics_advanced':False,'valid_steps':step+1,'valid_stop_reason':stop,'final_time_s':finaltime.tolist(),'wall_seconds':elapsed,'world_control_steps_per_second':4/elapsed,'limitations':'Host synchronizes before each physics step and aborts entire batch; this proof does not implement contact-force guard, independent world resets, policy, reward or training. Thresholds are assumptions, not hardware limits. Timing is one short run.'}
(p/'sector_warp_guarded_step_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
if stop:raise RuntimeError(stop)
