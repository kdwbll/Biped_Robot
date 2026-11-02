"""Compare a float32 Warp PI servo to CPU controller; no training or hardware."""
from pathlib import Path
from dataclasses import replace
import json,sys,time,traceback
import numpy as np,mujoco
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
import mujoco_warp as mjw

@wp.kernel
def update_targets(initial:wp.array2d(dtype=wp.float32),targets:wp.array2d(dtype=wp.float32),delta_us:float):
 w,j=wp.tid();change=wp.float32(0.)
 if j==4:change=wp.float32(delta_us/2000.)*wp.float32(4.71238898038469)
 desired=initial[w,j]+change;limit=wp.float32(1.0471975511965976/.26*.02)
 targets[w,j]+=wp.clamp(desired-targets[w,j],-limit,limit)

@wp.kernel
def servo_step(targets:wp.array2d(dtype=wp.float32),integral:wp.array2d(dtype=wp.float32),length:wp.array2d(dtype=wp.float32),velocity:wp.array2d(dtype=wp.float32),ctrl:wp.array2d(dtype=wp.float32),dt:float):
 w,j=wp.tid();kp=wp.float32(4.);ki=wp.float32(1.5);kd=wp.float32(.08)
 if j==4:kp=wp.float32(8.);ki=wp.float32(6.);kd=wp.float32(0.)
 error=targets[w,j]-length[w,j];vel=velocity[w,j];raw=kp*error-kd*vel+integral[w,j];speed=wp.float32(1.0471975511965976/.26);stall=wp.float32(24.*.0980665);available=stall
 if raw*vel>0.:available=stall*wp.max(0.,1.-wp.abs(vel)/speed)
 torque=wp.clamp(raw,-available,available);permit=wp.abs(raw)<=available or (raw-torque)*error<0.
 if permit:integral[w,j]=wp.clamp(integral[w,j]+ki*error*wp.float32(dt),-1.,1.)
 ctrl[w,j]=torque

out={'model':'diagnostic_sector_implicit_stiff_trial.xml','nworld':1,'training_started':False,'samples':[]}
try:
 wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/out['model']));m.opt.timestep=.00025
 c=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,c,0);mujoco.mj_forward(m,c)
 servo=CandidateStandingServo(m,c,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
 hips=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)];hip0=c.qpos[hips].copy()
 with wp.ScopedDevice('cuda:0'):
  gm=mjw.put_model(m);gd=mjw.put_data(m,c,nworld=1,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize()
  initial=wp.array(np.array([servo.initial_positions],dtype=np.float32),dtype=wp.float32,device='cuda:0');targets=wp.array(np.array([servo.initial_positions],dtype=np.float32),dtype=wp.float32,device='cuda:0');integral=wp.zeros((1,5),dtype=wp.float32,device='cuda:0')
  with wp.ScopedCapture() as capture:
   wp.launch(servo_step,dim=(1,5),inputs=[targets,integral,gd.actuator_length,gd.actuator_velocity,gd.ctrl,m.opt.timestep]);mjw.step(gm,gd)
  graph=capture.graph;start=time.perf_counter();stop=None
  for step in range(2000):
   if step>=800:servo.set_pulse_us([1500,1500,1500,1500,1510])
   if step%80==0:wp.launch(update_targets,dim=(1,5),inputs=[initial,targets,10. if step>=800 else 0.])
   old=c.time;servo.step();wp.capture_launch(graph)
   if not np.isfinite(c.qpos).all() or not np.isfinite(c.qvel).all() or c.time<old+.5*m.opt.timestep:stop='cpu nonfinite/reset';break
   if max(abs(np.rad2deg(c.qpos[hips]-hip0)))>4.5:stop='cpu hip range';break
   if step+1 in (1,80,400,800,1200,2000):
    wp.synchronize();q=gd.qpos.numpy()[0];v=gd.qvel.numpy()[0];overflow=gd.overflow.numpy();actual_hip=np.rad2deg(q[hips]-hip0)
    if not np.isfinite(q).all() or not np.isfinite(v).all() or np.any(overflow):stop='gpu nonfinite/overflow';break
    if max(abs(actual_hip))>4.5:stop='gpu sampled hip range';break
    r={'step':step+1,'time_s':float(c.time),'max_qpos_error':float(np.max(np.abs(q-c.qpos))),'max_qvel_error':float(np.max(np.abs(v-c.qvel))),'max_ctrl_error':float(np.max(np.abs(gd.ctrl.numpy()[0]-c.ctrl))),'cpu_hip_delta_deg':np.rad2deg(c.qpos[hips]-hip0).tolist(),'gpu_hip_delta_deg':actual_hip.tolist(),'overflow':overflow.tolist()};out['samples'].append(r);print(r,flush=True)
  wp.synchronize();n=int(gd.nacon.numpy()[0]);gg=gd.contact.geom.numpy()[:n];gids=wp.array(np.arange(n,dtype=np.int32),dtype=wp.int32,device='cuda:0');gforce=wp.zeros(n,dtype=wp.spatial_vector,device='cuda:0')
  if n:mjw.contact_force(gm,gd,gids,False,gforce);wp.synchronize()
  forces=gforce.numpy();ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');gpu=[];cp=[]
  for i,(a,b) in enumerate(gg):
   if ground in (a,b):gpu.append({'other':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(b if a==ground else a)),'normal_n':float(forces[i,0])})
  for i,contact in enumerate(c.contact[:c.ncon]):
   if ground in (contact.geom1,contact.geom2):
    force=np.zeros(6);mujoco.mj_contactForce(m,c,i,force);other=contact.geom2 if contact.geom1==ground else contact.geom1;cp.append({'other':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other),'normal_n':float(force[0])})
  out['ground_forces']={'gpu':gpu,'cpu':cp,'gpu_total_n':sum(x['normal_n'] for x in gpu),'cpu_total_n':sum(x['normal_n'] for x in cp),'gpu_contact_count':n};print('ground forces',out['ground_forces'],flush=True)
  out['stop_reason']=stop;out['wall_seconds_including_cpu_and_sync']=time.perf_counter()-start;out['success']=stop is None
except Exception as e:out.update({'success':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
out['limitations']='One GPU world, 0.5 s, single 10 us central step and ground force extraction. No force guard, no batch, no domain randomization and no training; GPU hip checked only at samples. Numerical divergence may reflect contact dynamics, precision or controller differences; not parity certification.'
(p/'sector_warp_contact_force_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('success',out['success'],flush=True)
