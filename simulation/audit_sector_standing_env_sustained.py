"""Exercise standing env under sustained center commands, without training."""
from pathlib import Path
import json
import mujoco
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;rows=[]
for command in (-1.,1.):
 env=SectorStandingEnv(episode_seconds=4.);env.reset(seed=1);ground=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'ground');max_force=0.;peak_pair=None;maxhip=0.;internal_peak=0.;gear_peak=0.
 for i in range(200):
  obs,reward,term,trunc,info=env.step(np.array([0,0,0,0,command]));maxhip=max(maxhip,info.get('max_hip_deg',0.));internal_peak=max(internal_peak,info.get('sampled_peak_self_contact_n',0.));gear_peak=max(gear_peak,info.get('max_gear_residual_deg',0.))
  for j,c in enumerate(env.data.contact[:env.data.ncon]):
   if ground in (c.geom1,c.geom2):continue
   f=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,j,f)
   if f[0]>max_force:max_force=float(f[0]);peak_pair=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  if term or trunc:break
 row={'action':command,'steps':i+1,'terminated':term,'truncated':trunc,'reason':info.get('reason'),'simulation_time':info.get('simulation_time'),'max_hip_change_deg':maxhip,'env_sampled_peak_self_contact_n':internal_peak,'max_gear_residual_deg':gear_peak,'sampled_peak_self_contact_n':max_force,'sampled_peak_pair':peak_pair}
 rows.append(row);print(row,flush=True)
out={'results':rows,'limitations':'Two four-second sustained actions, sampled contacts once per control frame. Not training, full collision validation, or hardware safety.'}
(p/'sector_standing_env_sustained_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
