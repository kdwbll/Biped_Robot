"""Evaluate a local standing checkpoint against zero action, without training."""
from pathlib import Path
import argparse
import json
from stable_baselines3 import PPO
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
from train_guarded_standing import evaluate

parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args();model=PPO.load(args.model,device='cpu')
env=SectorWarpStandingVecEnv(nworld=8,episode_seconds=3.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped')
try:
    rows=[]
    for seed in (9040,9060):
        print(f'EVALUATING seed {seed}: zero action',flush=True);zero=evaluate(env,None,seed)
        print(f'EVALUATING seed {seed}: trained policy',flush=True);trained=evaluate(env,model,seed)
        rows.append({'seed':seed,'zero_action':zero,'trained_policy':trained})
    result={'results':rows,'limitations':'Two evaluation seeds, 8 worlds each, 3 seconds per world; assumed dynamics and no walking/hardware validation.'}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
finally:env.close()
