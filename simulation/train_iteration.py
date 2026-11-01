"""Reproducible short PPO runs; always exploratory, never a hardware controller."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from randomized_biped_env import PerturbedBipedEnv
from alternating_biped_env import AlternatingBipedEnv
from larger_action_env import LargerActionBipedEnv
from clocked_biped_env import ClockedBipedEnv
from safer_clocked_env import SaferClockedEnv
from stable_baselines3.common.callbacks import CheckpointCallback

ROOT=Path(__file__).parent
EVAL_SEEDS=(7001,7002,7003,7004,7005)
def evaluate(policy,reward='velocity'):
    rows=[]
    for seed in EVAL_SEEDS:
        env=SaferClockedEnv(episode_seconds=4,disturbance=.02) if reward=='safer_clocked' else ClockedBipedEnv(episode_seconds=4,disturbance=.02) if reward=='clocked' else LargerActionBipedEnv(episode_seconds=4,disturbance=.02) if reward=='larger_action' else AlternatingBipedEnv(episode_seconds=4,disturbance=.02);obs,_=env.reset(seed=seed);x0=float(env.data.xipos[env.body,0]);max_tilt=0.;ret=0.;steps=0
        while True:
            action=np.zeros(5,dtype=np.float32) if policy is None else policy.predict(obs,deterministic=True)[0]
            obs,r,term,trunc,info=env.step(action);ret+=r;steps+=1
            max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
            if term or trunc:break
        rows.append({'seed':seed,'seconds':round(steps*.02,3),'distance_m':float(env.data.xipos[env.body,0]-x0),'max_tilt_deg':max_tilt,'fell':bool(term),'support_switches':int(info.get('support_switches',0)),'reward':float(ret)});env.close()
    return rows

def env_factory(reward):
    def make():return SaferClockedEnv(episode_seconds=4,disturbance=.02) if reward=='safer_clocked' else ClockedBipedEnv(episode_seconds=4,disturbance=.02) if reward=='clocked' else LargerActionBipedEnv(episode_seconds=4,disturbance=.02) if reward=='larger_action' else AlternatingBipedEnv(episode_seconds=4,disturbance=.02) if reward=='alternation' else PerturbedBipedEnv(episode_seconds=4,disturbance=.02)
    return make

def main():
    a=argparse.ArgumentParser();a.add_argument('--reward',choices=['velocity','alternation','larger_action','clocked','safer_clocked'],default='velocity');a.add_argument('--steps',type=int,default=8192);a.add_argument('--workers',type=int,default=4);a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--name',required=True);opt=a.parse_args()
    if opt.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);print('training configuration',vars(opt),flush=True)
    v=DummyVecEnv([env_factory(opt.reward)]) if opt.workers==1 else SubprocVecEnv([env_factory(opt.reward) for _ in range(opt.workers)],start_method='spawn')
    model=PPO('MlpPolicy',v,seed=42,device=opt.device,n_steps=256,batch_size=256,n_epochs=4,learning_rate=.0003,verbose=1)
    start=time.perf_counter(); checkpoint=CheckpointCallback(save_freq=max(1,4096//opt.workers),save_path=str(ROOT/'training_checkpoints'),name_prefix=opt.name);model.learn(total_timesteps=opt.steps,callback=checkpoint);duration=time.perf_counter()-start
    target=ROOT/(opt.name+'.zip');model.save(str(target));v.close();print('model_saved',str(target),'elapsed_seconds',round(duration,2),flush=True)
    result={'configuration':vars(opt),'train_seconds':duration,'neutral':evaluate(None,opt.reward),'policy':evaluate(model,opt.reward)}
    (ROOT/(opt.name+'_evaluation.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8');print('evaluation',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()



