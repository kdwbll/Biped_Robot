"""Verify an untrained PPO policy can use the candidate environment; no learning."""
from pathlib import Path
import json
import numpy as np
from stable_baselines3 import PPO
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped')
try:
    env.seed(314);obs=env.reset()
    model=PPO('MlpPolicy',env,device='cpu',n_steps=8,batch_size=8,verbose=0)
    actions,_=model.predict(obs,deterministic=True)
    obs2,rewards,done,infos=env.step(actions)
    if actions.shape!=(4,5) or not np.isfinite(obs2).all() or not np.isfinite(rewards).all():raise RuntimeError('Invalid policy interface')
    out={'worlds':4,'action_shape':list(actions.shape),'reward_shape':list(rewards.shape),
         'finite':True,'done':done.tolist(),'guard_flags':[i['guard_flags'] for i in infos],
         'learn_called':False,'saved_model':False,'physics':'Warp CUDA','policy_device':str(model.device),
         'limitations':'One untrained policy inference and one step only; no policy optimization or evidence of walking.'}
    (p/'sector_shaped_policy_interface_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:env.close()
