"""Inspect why a local gait candidate trips simulation guards; no learning."""
from pathlib import Path
import argparse,json,collections
from stable_baselines3 import PPO
from sector_gait_curriculum import GaitCurriculum
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--output',required=True);args=a.parse_args()
model=PPO.load(args.model,device='cpu');env=GaitCurriculum(SectorWarpStandingVecEnv(nworld=16,episode_seconds=4.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=2.,action_amplitude_us=[150]*4+[20.]),objective='load_shift',load_weight=1.)
try:
 env.seed(9901);obs=env.reset();events=[]
 for step in range(200):
  actions,_=model.predict(obs,deterministic=True);obs,_,done,infos=env.step(actions)
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags']:
    events.append({'step':step,'world':i,'flags':info['guard_flags'],'peak_force_n':info['sampled_peak_self_contact_n'],
                   'peak_pair':info['sampled_peak_self_contact_pair'],'action':actions[i].tolist()})
 counts=dict(collections.Counter(str(e['flags']) for e in events));pairs=dict(collections.Counter(str(e['peak_pair']) for e in events))
 out={'events':events,'flag_counts':counts,'peak_pair_counts':pairs,'limitations':'One seed, 16 worlds, four seconds; sampled contact forces and heuristic guards, no training or hardware.'}
 Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'event_count':len(events),'flag_counts':counts,'peak_pair_counts':pairs},indent=2),flush=True)
finally:env.close()
