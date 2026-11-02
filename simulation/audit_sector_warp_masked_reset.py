"""Validate masked reset of one Warp world; no policy or training."""
from pathlib import Path
from dataclasses import replace
import json,sys
import numpy as np,mujoco
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
import mujoco_warp as mjw
from sector_warp_guard import restore_masked_state,scan_state
from sector_warp_servo_kernels import servo_step
wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));m.opt.timestep=.00025;c=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,c,0);mujoco.mj_forward(m,c)
s=CandidateStandingServo(m,c,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,c,nworld=4,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize()
 targets=wp.array(np.tile(s.initial_positions.astype(np.float32),(4,1)),dtype=wp.float32,device='cuda:0');integral=wp.zeros((4,5),dtype=wp.float32,device='cuda:0')
 with wp.ScopedCapture() as cap:
  wp.launch(servo_step,dim=(4,5),inputs=[targets,integral,gd.actuator_length,gd.actuator_velocity,gd.ctrl,m.opt.timestep,0.]);mjw.step(gm,gd)
 for _ in range(80):wp.capture_launch(cap.graph)
 wp.synchronize();oldq=gd.qpos.numpy().copy();oldv=gd.qvel.numpy().copy();oldt=gd.time.numpy().copy()
 # Synthetic nonzero integrator/flag confirms the selected state is cleared.
 integ=np.ones((4,5),dtype=np.float32);wp.copy(integral,wp.array(integ,dtype=wp.float32,device='cuda:0'));flags=wp.zeros(4,dtype=wp.int32,device='cuda:0')
 hips=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)];shaft=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')])
 injected=oldq.copy();injected[1,hips[0]]+=np.deg2rad(4.6);wp.copy(gd.qpos,wp.array(injected,dtype=wp.float32,device='cuda:0'))
 scan=[gd.qpos,gd.qvel,flags,m.nq,m.nv,hips[0],hips[1],shaft,float(c.qpos[hips[0]]),float(c.qpos[hips[1]]),float(c.qpos[shaft]),float(np.deg2rad(4.5)),float(np.deg2rad(.2))]
 wp.launch(scan_state,dim=4,inputs=scan);wp.synchronize();detected=flags.numpy()
 if detected.tolist()!=[0,6,0,0]:raise RuntimeError(('injected guard not detected',detected.tolist()))
 mask=wp.array(np.array([False,True,False,False]),dtype=wp.bool,device='cuda:0');initial_q=wp.array(c.qpos.astype(np.float32),dtype=wp.float32,device='cuda:0');initial_target=wp.array(s.initial_positions.astype(np.float32),dtype=wp.float32,device='cuda:0')
 mjw.reset_data(gm,gd,mask);wp.launch(restore_masked_state,dim=4,inputs=[gd.qpos,gd.qvel,gd.ctrl,targets,integral,flags,mask,initial_q,initial_target,m.nq,m.nv,m.nu]);mjw.forward(gm,gd);wp.synchronize()
 q=gd.qpos.numpy();v=gd.qvel.numpy();times=gd.time.numpy();ft=flags.numpy();it=integral.numpy();tg=targets.numpy()
 keep=[0,2,3];other_q=float(np.max(np.abs(q[keep]-oldq[keep])));other_v=float(np.max(np.abs(v[keep]-oldv[keep])));selected_q=float(np.max(np.abs(q[1]-c.qpos)));selected_v=float(np.max(np.abs(v[1])));other_t=float(np.max(np.abs(times[keep]-oldt[keep])))
 if max(other_q,other_v,other_t)>1e-7 or selected_q>1e-6 or selected_v>1e-7 or abs(times[1])>1e-7 or ft.tolist()!=[0,0,0,0] or np.any(it[1]!=0) or not np.allclose(it[keep],1):raise RuntimeError('Masked reset failed')
out={'worlds':4,'reset_world':1,'detected_guard_flags_before_reset':detected.tolist(),'unselected_max_q_change':other_q,'unselected_max_v_change':other_v,'unselected_max_time_change':other_t,'reset_q_max_error':selected_q,'reset_v_max_abs':selected_v,'reset_time_s':float(times[1]),'flags_after':ft.tolist(),'limitations':'One masked reset following an injected out-of-range joint state after 0.02 s, not a rollout or vector training environment. Does not yet validate changing per-world friction or reset immediately after actual overflow/nonfinite.'}
(p/'sector_warp_masked_reset_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
