"""Train a phase-conditioned foot-unloading candidate; simulation only."""
from pathlib import Path
import argparse,json,time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
from sector_gait_curriculum import GaitCurriculum

class Checkpoints(BaseCallback):
    def __init__(self,path):super().__init__();self.path=path;self.start=time.monotonic();self.events=0;self.falls=0
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done and info['guard_flags']:self.events+=1
            elif done and not info['TimeLimit.truncated']:self.falls+=1
        return True
    def _on_rollout_end(self):
        row={'timesteps':self.num_timesteps,'elapsed_s':time.monotonic()-self.start,'guard_terminations':self.events,'falls':self.falls}
        with (self.path/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        if self.num_timesteps%4096==0:self.model.save(str(self.path/f'checkpoint_pre_update_{self.num_timesteps}'))
        print('PROGRESS '+json.dumps(row),flush=True)

def evaluate(env,model,seed):
    env.seed(seed);obs=env.reset();n=env.num_envs;air=np.zeros(2);stance_air=0;matched=0;valid=0;guards=0;falls=0;timeouts=0;alignment=0.;returns=np.zeros(n)
    for step in range(200):
        action=np.zeros((n,5)) if model is None else model.predict(obs,deterministic=True)[0]
        obs,reward,done,infos=env.step(action);returns+=reward
        for i,info in enumerate(infos):
            if done[i]:
                if info['guard_flags']:guards+=1
                elif info['TimeLimit.truncated']:timeouts+=1
                else:falls+=1
            if info['guard_flags']:continue
            contact=np.asarray(info['curriculum_sampled_contact']);swing=info['curriculum_swing_leg'];air+=~contact
            matched+=int((not contact[swing]) and contact[1-swing]);stance_air+=int(not contact[1-swing]);alignment+=info['curriculum_load_alignment'];valid+=1
    return {'seed':seed,'worlds':n,'frames':200,'mean_return':float(np.mean(returns)),
            'air_fraction':(air/max(1,valid)).tolist(),'phase_matched_swing_fraction':matched/max(1,valid),
            'stance_air_fraction':stance_air/max(1,valid),'mean_load_alignment':alignment/max(1,valid),'guard_terminations':guards,'falls':falls,'timeouts':timeouts}

def main():
    a=argparse.ArgumentParser();a.add_argument('--output-dir',required=True);a.add_argument('--timesteps',type=int,default=16384);a.add_argument('--objective',choices=['unload','load_shift'],default='unload');a.add_argument('--worlds',type=int,default=16);a.add_argument('--load-weight',type=float,default=.5);a.add_argument('--risk-weight',type=float,default=0.);a.add_argument('--leg-amplitude',type=float,default=150.);a.add_argument('--resume');args=a.parse_args()
    p=Path(args.output_dir);p.mkdir(parents=True,exist_ok=False)
    config={'worlds':args.worlds,'timesteps':args.timesteps,'episode_seconds':4.,'reset_tilt_deg':1.,'leg_amplitude_us':args.leg_amplitude,'slew_us_per_frame':2.,'phase_period_s':1.6,'load_weight':args.load_weight,'risk_weight':args.risk_weight,
            'objective':args.objective,'resume_checkpoint':args.resume,'scope':'hypothetical mechanics, phase-conditioned load shift or foot unloading; not walking or hardware'}
    (p/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    base=SectorWarpStandingVecEnv(nworld=args.worlds,episode_seconds=4.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=2.,action_amplitude_us=[args.leg_amplitude]*4+[20.])
    env=GaitCurriculum(base,objective=args.objective,load_weight=args.load_weight,contact_risk_weight=args.risk_weight)
    try:
        model=PPO.load(args.resume,env=env,device='cpu') if args.resume else PPO('MlpPolicy',env,device='cpu',seed=2211,n_steps=64,batch_size=256,n_epochs=4,verbose=1)
        print('BASELINE ZERO',flush=True);zero=evaluate(env,None,9901)
        (p/'baseline.json').write_text(json.dumps(zero,indent=2),encoding='utf8')
        print('TRAINING START',flush=True);env.seed(2211);model.learn(total_timesteps=args.timesteps,callback=Checkpoints(p),reset_num_timesteps=not bool(args.resume))
        model.save(str(p/'final_policy'));print('EVALUATION TRAINED',flush=True);trained=evaluate(env,model,9901)
        out={'config':config,'baseline':zero,'trained':trained,'actual_timesteps':model.num_timesteps,
             'limitations':'One short deterministic evaluation on sampled contact proxy; no true foot-height validation, walking or hardware.'}
        (p/'result.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('RESULT '+json.dumps(out),flush=True)
    except BaseException as e:
        (p/'failure.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__},indent=2),encoding='utf8');raise
    finally:env.close()
if __name__=='__main__':main()
