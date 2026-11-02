"""Short experimental GPU-physics PPO standing run; not a hardware controller."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

class Progress(BaseCallback):
    def __init__(self,directory):
        super().__init__();self.directory=directory;self.guard=0;self.fall=0;self.timeout=0;self.start=time.monotonic()
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                if info['guard_flags']:self.guard+=1
                elif info['TimeLimit.truncated']:self.timeout+=1
                else:self.fall+=1
        return True
    def _on_rollout_end(self):
        record={'timesteps':self.num_timesteps,'wall_seconds':time.monotonic()-self.start,
                'guard_terminations':self.guard,'falls':self.fall,'timeouts':self.timeout}
        with (self.directory/'progress.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(record)+'\n')
        self.model.save(str(self.directory/'latest_before_update'))
        print('PROGRESS '+json.dumps(record),flush=True)

def evaluate(env,policy,seed,frames=150):
    env.seed(seed);obs=env.reset();n=env.num_envs;total=np.zeros(n);guard=fall=timeout=0;peak=0.;minimum=1.;max_tilt=0.
    for _ in range(frames):
        action=np.zeros((n,5)) if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,reward,done,infos=env.step(action);total+=reward
        if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise RuntimeError('Nonfinite evaluation')
        for end,info in zip(done,infos):
            peak=max(peak,info['sampled_peak_self_contact_n'])
            if not info['guard_flags']:
                minimum=min(minimum,info['upright']);max_tilt=max(max_tilt,float(np.rad2deg(np.arccos(np.clip(info['upright'],-1,1)))))
            if end:
                if info['guard_flags']:guard+=1
                elif info['TimeLimit.truncated']:timeout+=1
                else:fall+=1
    return {'seed':seed,'worlds':n,'frames':frames,'sim_seconds_per_world':frames*.02,'mean_return':float(total.mean()),
            'guard_terminations':guard,'falls':fall,'timeouts':timeout,'minimum_upright':minimum,
            'max_tilt_deg':max_tilt,'max_sampled_self_contact_n':peak}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);parser.add_argument('--timesteps',type=int,default=4096)
    args=parser.parse_args();directory=Path(args.output_dir);directory.mkdir(parents=True,exist_ok=False)
    config={'timesteps_requested':args.timesteps,'worlds':16,'episode_seconds':3.,'reset_tilt_deg':1.,'reward_mode':'shaped',
            'physics':'MuJoCo Warp CUDA','policy':'SB3 PPO CPU','train_seed':2026,'evaluation_seed':9026,
            'contact_guard_n':20.,'gear_guard_deg':.2,'n_steps':64,'batch_size':256,'n_epochs':4,
            'scope':'experimental simulation-only standing; no walking training or physical deployment'}
    write_json(directory/'config.json',config)
    env=SectorWarpStandingVecEnv(nworld=16,episode_seconds=3.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped')
    try:
        model=PPO('MlpPolicy',env,device='cpu',seed=2026,n_steps=64,batch_size=256,n_epochs=4,verbose=1)
        print('BASELINE zero action',flush=True);zero=evaluate(env,None,9026)
        print('BASELINE untrained policy',flush=True);before=evaluate(env,model,9026)
        write_json(directory/'baseline.json',{'zero_action':zero,'untrained_policy':before})
        print('TRAINING START',flush=True);env.seed(2026);model.learn(total_timesteps=args.timesteps,callback=Progress(directory),progress_bar=False)
        model.save(str(directory/'standing_final'))
        print('EVALUATION trained policy',flush=True);after=evaluate(env,model,9026)
        result={'config':config,'zero_action':zero,'untrained_policy':before,'trained_policy':after,
                'training_completed':True,'actual_timesteps':int(model.num_timesteps),
                'limitations':'Short deterministic evaluation on a single seed and assumed physical model; no walking or hardware validation.'}
        write_json(directory/'result.json',result);print('RESULT '+json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(directory/'failure.json',{'error_type':type(error).__name__,'message':str(error),'timesteps':int(model.num_timesteps) if 'model' in locals() else 0})
        raise
    finally:env.close()

if __name__=='__main__':main()
