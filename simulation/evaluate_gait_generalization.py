"""Compare a gait curriculum checkpoint to zero action on held-out seeds."""
from pathlib import Path
import argparse,json
import numpy as np
from stable_baselines3 import PPO
from sector_gait_curriculum import GaitCurriculum
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

def evaluate(env,model,seed):
 env.seed(seed);obs=env.reset();base=env.venv;n=env.num_envs;verts=[]
 for geom in base.soles:
  mesh=int(base.m.geom_dataid[geom]);start=int(base.m.mesh_vertadr[mesh]);count=int(base.m.mesh_vertnum[mesh]);verts.append(base.m.mesh_vert[start:start+count].copy())
 total=np.zeros(n);air=np.zeros(2);matched=0;valid=0;guards=falls=timeouts=0;align=0.;max_clear=np.full(2,-1.);clear_frames=np.zeros(2,int);samples=0
 for step in range(200):
  action=np.zeros((n,5)) if model is None else model.predict(obs,deterministic=True)[0]
  obs,reward,done,infos=env.step(action);total+=reward
  for i,info in enumerate(infos):
   if done[i]:
    if info['guard_flags']:guards+=1
    elif info['TimeLimit.truncated']:timeouts+=1
    else:falls+=1
   if info['guard_flags']:continue
   contact=np.asarray(info['curriculum_sampled_contact']);swing=info['curriculum_swing_leg'];air+=~contact
   matched+=int((not contact[swing]) and contact[1-swing]);align+=info['curriculum_load_alignment'];valid+=1
  if step%5==0:
   positions=base.gd.geom_xpos.numpy()[:,base.soles];rotations=base.gd.geom_xmat.numpy()[:,base.soles].reshape(n,2,3,3)
   for leg in (0,1):
    z=positions[:,leg,2,None]+rotations[:,leg,2,:]@verts[leg].T;lowest=z.min(axis=1)
    active=np.array([not info['guard_flags'] for info in infos]);max_clear[leg]=max(max_clear[leg],float(np.max(lowest[active],initial=-1.)))
    clear_frames[leg]+=int(np.sum((lowest>.001)&active))
   samples+=n
 return {'seed':seed,'mean_return':float(np.mean(total)),'guard_terminations':guards,'falls':falls,'timeouts':timeouts,
         'air_fraction':(air/max(1,valid)).tolist(),'phase_matched_swing_fraction':matched/max(1,valid),
         'mean_load_alignment':align/max(1,valid),'sampled_max_sole_clearance_m':max_clear.tolist(),
         'sampled_clearance_over_1mm_fraction':(clear_frames/max(samples,1)).tolist(),'clearance_sampling_s':.1}

a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--output',required=True);a.add_argument('--leg-amplitude',type=float,default=150.);a.add_argument('--risk-weight',type=float,default=1.);args=a.parse_args()
model=PPO.load(args.model,device='cpu')
env=GaitCurriculum(SectorWarpStandingVecEnv(nworld=8,episode_seconds=4.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=2.,action_amplitude_us=[args.leg_amplitude]*4+[20.]),objective='load_shift',load_weight=1.,contact_risk_weight=args.risk_weight)
try:
 rows=[]
 for seed in (9911,9921,9931):
  print(f'SEED {seed}: zero',flush=True);zero=evaluate(env,None,seed)
  print(f'SEED {seed}: trained',flush=True);trained=evaluate(env,model,seed)
  rows.append({'seed':seed,'zero_action':zero,'trained_policy':trained})
 out={'leg_action_amplitude_us':args.leg_amplitude,'risk_weight':args.risk_weight,'results':rows,'limitations':'Three seeds, eight worlds and four seconds per evaluation; ground contacts are sampled and dynamics unmeasured. Not evidence of walking or hardware safety.'}
 Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
