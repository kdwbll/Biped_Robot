"""Check SB3 policy can call experimental Warp VecEnv without learning."""
from pathlib import Path
import json,sys
import numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import torch
from stable_baselines3 import PPO
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,contact_limit_n=20.,gear_limit_deg=.2,pwm_slew_us=1.)
try:
 model=PPO('MlpPolicy',env,n_steps=16,batch_size=64,n_epochs=1,device='cpu',seed=1,verbose=0)
 obs=env.reset();actions,_=model.predict(obs,deterministic=True)
 if actions.shape!=(4,5) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1):raise RuntimeError('Invalid policy output')
 obs2,reward,done,info=env.step(actions)
 if obs2.shape!=(4,16) or not np.isfinite(obs2).all() or not np.isfinite(reward).all():raise RuntimeError('Invalid environment output')
 out={'worlds':4,'policy_action_shape':list(actions.shape),'observation_shape':list(obs2.shape),'finite':True,'learn_called':False,'saved_model':False,'policy_device':'cpu','torch_cuda_available':bool(torch.cuda.is_available()),'physics_backend':'MuJoCo Warp cuda:0','done':done.tolist(),'limitations':'One untrained policy inference and one environment step; no PPO optimization, learning, reward validation, multi-episode robustness or hardware validation.'}
 (p/'sector_warp_sb3_wiring_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
