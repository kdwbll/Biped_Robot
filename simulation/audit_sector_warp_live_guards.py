"""Test host-synchronized GPU state and sampled force guard on real simulated contacts."""
from pathlib import Path
from dataclasses import replace
import json,sys,time,traceback
import mujoco,numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
import mujoco_warp as mjw
from sector_warp_guard import scan_state,scan_contacts
from sector_warp_servo_kernels import servo_step,update_targets
wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();results=[]
for filename,kd,nsteps in [('diagnostic_sector_implicit_stiff_trial.xml',0.,80),('diagnostic_sector_collision_trial.xml',.01,16000)]:
 m=mujoco.MjModel.from_xml_path(str(p/filename));m.opt.timestep=.00025;c=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,c,0);mujoco.mj_forward(m,c)
 servo=CandidateStandingServo(m,c,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=kd)
 h=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)];shaft=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')]);ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 with wp.ScopedDevice('cuda:0'):
  gm=mjw.put_model(m);gd=mjw.put_data(m,c,nworld=4,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize()
  flags=wp.zeros(4,dtype=wp.int32,device='cuda:0');args=[gd.qpos,gd.qvel,flags,m.nq,m.nv,h[0],h[1],shaft,float(c.qpos[h[0]]),float(c.qpos[h[1]]),float(c.qpos[shaft]),float(np.deg2rad(4.5)),float(np.deg2rad(.2))]
  init=wp.array(np.tile(np.asarray(servo.initial_positions,dtype=np.float32),(4,1)),dtype=wp.float32,device='cuda:0');targets=wp.array(np.tile(np.asarray(servo.initial_positions,dtype=np.float32),(4,1)),dtype=wp.float32,device='cuda:0');integ=wp.zeros((4,5),dtype=wp.float32,device='cuda:0')
  with wp.ScopedCapture() as cap:
   wp.launch(servo_step,dim=(4,5),inputs=[targets,integ,gd.actuator_length,gd.actuator_velocity,gd.ctrl,m.opt.timestep,kd]);mjw.step(gm,gd)
  graph=cap.graph;stop=None;start=time.perf_counter();previous=gd.time.numpy().copy();last_peak=0.;last_pair=None;steps_done=0
  for step in range(nsteps):
   wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();bits=flags.numpy();overflow=gd.overflow.numpy();now=gd.time.numpy()
   if np.any(bits) or np.any(overflow):stop='state_or_overflow';break
   if step and np.any(now<previous+.5*m.opt.timestep):stop='time_reset';break
   previous=now.copy()
   if step%80==0:wp.launch(update_targets,dim=(4,5),inputs=[init,targets,-20. if step>=4000 else 0.])
   wp.capture_launch(graph);steps_done+=1
   if step%10==0:
    wp.synchronize();n=int(gd.nacon.numpy()[0])
    if n:
     ids=wp.array(np.arange(n,dtype=np.int32),dtype=wp.int32,device='cuda:0');forces=wp.zeros(n,dtype=wp.spatial_vector,device='cuda:0');mjw.contact_force(gm,gd,ids,False,forces);wp.launch(scan_contacts,dim=n,inputs=[gd.contact.geom,gd.contact.worldid,forces,flags,n,4,ground,20.]);wp.synchronize()
     force=forces.numpy();geom=gd.contact.geom.numpy()[:n];nonfloor=(geom[:,0]!=ground)&(geom[:,1]!=ground)
     if nonfloor.any():
      ix=np.flatnonzero(nonfloor)[np.argmax(force[nonfloor,0])]
      if force[ix,0]>last_peak:last_peak=float(force[ix,0]);last_pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in geom[ix]]
     if np.any(flags.numpy()):stop='contact_guard';break
  wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();bits=flags.numpy();overflow=gd.overflow.numpy();end=gd.time.numpy()
  if stop is None and (np.any(bits) or np.any(overflow)):stop='state_or_overflow'
  if stop is None and steps_done and np.any(end<previous+.5*m.opt.timestep):stop='time_reset'
  row={'model':filename,'steps':steps_done,'sim_time_s':end.tolist(),'stop_reason':stop,'flags':bits.tolist(),'overflow':overflow.tolist(),'sampled_peak_self_contact_n':last_peak,'sampled_peak_pair':last_pair,'wall_seconds':time.perf_counter()-start};results.append(row);print(row,flush=True)
out={'results':results,'heuristic_limit_n':20.,'force_sample_interval_s':.0025,'limitations':'Host-synchronized prototype aborts entire batch; no per-world reset, training, PPO or reward. Threshold is arbitrary and sampled forces can miss spikes; float32 vs CPU and long-run parity remain unverified. Not hardware safety.'}
(p/'sector_warp_live_guards_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8')
