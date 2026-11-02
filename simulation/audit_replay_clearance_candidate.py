"""Replay a discovered open-loop clearance candidate with support and tilt checks."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;candidate=json.loads((p/'exploration_clearance_audit.json').read_text())['best'];world=candidate['world'];history=np.asarray(candidate['action_history'])
env=SectorWarpStandingVecEnv(nworld=1,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.seed(4012+world);env.reset();verts=[]
 for geom in env.soles:
  mesh=int(env.m.geom_dataid[geom]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[start:start+count].copy())
 frames=[];guard=None
 for step in range(min(candidate['step']+15,len(history)*30)):
  _,_,done,infos=env.step(history[step//30][None,:]);info=infos[0]
  if done[0] and info['guard_flags']:guard={'step':step,'flags':info['guard_flags'],'pair':info['sampled_peak_self_contact_pair'],'force_n':info['sampled_peak_self_contact_n']};break
  pos=env.gd.geom_xpos.numpy()[0,env.soles];rot=env.gd.geom_xmat.numpy()[0,env.soles].reshape(2,3,3)
  heights=[float(np.min(pos[j,2]+verts[j]@rot[j,2,:])) for j in (0,1)]
  frames.append({'step':step,'sole_clearance_m':heights,'sampled_sole_peak_n':info['sampled_sole_peak_normal_n'],'upright':info['upright'],'height':info['height']})
 leg=candidate['leg'];supported=[f for f in frames if f['sole_clearance_m'][leg]>.001 and f['sampled_sole_peak_n'][1-leg]>.5 and f['upright']>np.cos(np.deg2rad(10))]
 out={'candidate':candidate,'frames':frames,'guard':guard,'supported_clearance_frames':len(supported),'max_supported_clearance_m':max((f['sole_clearance_m'][leg] for f in supported),default=0.),
      'limitations':'Single-world open-loop replay and 20 ms sampling with approximate model; does not establish sustained balance, gait or hardware safety.'}
 (p/'replay_clearance_candidate_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:out[k] for k in ('guard','supported_clearance_frames','max_supported_clearance_m')},indent=2),flush=True)
finally:env.close()
