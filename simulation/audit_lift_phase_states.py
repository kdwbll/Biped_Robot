"""Record live phase boundaries and terminal observations for alternating lifts."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
p=Path(__file__).parent
h0=np.asarray(json.loads((p/'训练记录/opposite_lift_20261103_a/best.json').read_text(encoding='utf8'))['action_history'])
h1=np.asarray(json.loads((p/'训练记录/lift_search_20261102_a/best.json').read_text(encoding='utf8'))['action_history'])
n=8;env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=6.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset_rng=[np.random.default_rng(12201+i) for i in range(n)];env.reset()
 alive=np.ones(n,bool);records=[[] for _ in range(n)];guards=[]
 def capture(frame,infos=None):
  q=env.gd.qpos.numpy();v=env.gd.qvel.numpy();length=env.gd.actuator_length.numpy();targets=env.targets.numpy()
  for i in np.flatnonzero(alive):
   records[i].append({'frame':frame,'root_position':q[i,:3].tolist(),'root_quaternion':q[i,3:7].tolist(),'root_velocity':v[i,:6].tolist(),'actuator_length':length[i].tolist(),'actuator_target':targets[i].tolist(),'pwm_us':env.pwm[i].tolist(),'sampled_sole_peak_normal_n':None if infos is None else infos[i]['sampled_sole_peak_normal_n']})
 capture(0)
 for step in range(270):
  phase=step//90;action=np.tile([h0,h1,h0][phase][(step%90)//30],(n,1));_,_,done,infos=env.step(action)
  for i in range(n):
   if alive[i] and done[i]:
    info=infos[i];terminal=info.get('terminal_observation');guards.append({'world':i,'frame':step+1,'guard_flags':int(info['guard_flags']),'sampled_peak_self_contact_pair':info['sampled_peak_self_contact_pair'],'sampled_peak_self_contact_n':info['sampled_peak_self_contact_n'],'terminal_observation':None if terminal is None else np.asarray(terminal).tolist()});alive[i]=False
  if (step+1)%90==0:capture(step+1,infos)
 out={'worlds':n,'order':'leg0_leg1_leg0','phase_boundary_records':records,'terminations':guards,'limitations':'Full states recorded only for live worlds after each phase; guard worlds are auto-reset by env, so no full terminal state is recorded. The sampled peak pair is diagnostic and may not be the cause of every guard. Terminal observation is limited. Sampled contact and end-of-frame clearance are not necessarily simultaneous; no hardware validation.'}
 (p/'训练记录/lift_phase_state_diagnostic_20261103.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'live_worlds_at_boundary':[sum(any(r['frame']==f for r in row) for row in records) for f in (0,90,180,270)],'terminations':guards},indent=2),flush=True)
finally:env.close()
