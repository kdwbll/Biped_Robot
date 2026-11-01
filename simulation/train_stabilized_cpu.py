"""Fresh experimental CPU baseline with numerically stabilized fifth servo."""
import json,os,time
from functools import partial
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from train_continuous import AdaptiveEnv,grade,gate,configure,save_json,save_model
from residual_gait_env import ResidualGaitEnv
from stabilized_servo import StabilizedServo
from live_telemetry import open_buffer
ROOT=Path(__file__).parent;STATE=ROOT/'stabilized_state.json';CURRENT=ROOT/'stabilized_current.zip';BEST=ROOT/'stabilized_best.zip'

class StableEnv(AdaptiveEnv):
 def reset(self,*,seed=None,options=None):
  _,info=super().reset(seed=seed,options=options);self.controller=StabilizedServo(self.model,self.data)
  return self._observation(),info
def make(slot):return StableEnv(slot)

def evaluate(model,weight):
 rows=[]
 for seed in (9101,9102,9103):
  e=ResidualGaitEnv(episode_seconds=4.,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
  obs,_=e.reset(seed=seed);e.controller=StabilizedServo(e.model,e.data);obs=e._observation()
  x0=e.data.xipos[e.body,0];run=[0,0];order=[];peak=[0.,0.]
  for k in range(200):
   action,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(action)
   for leg in (0,1):
    peak[leg]=max(peak[leg],info['foot_clearance_m'][leg]);swing=info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg]
    run[leg]=run[leg]+1 if swing else 0
    if run[leg]==3:order.append(leg)
   if term or trunc:break
  rows.append(dict(seed=seed,seconds=(k+1)*.02,fell=bool(term),forward_m=float(e.data.xipos[e.body,0]-x0),
   peak_clearance_m=peak,events=order,alternations=sum(order[i]!=order[i-1] for i in range(1,len(order)))))
  e.close()
 return rows

if __name__=='__main__':
 torch.set_num_threads(1);mapping,arr=open_buffer();arr[:]=0.
 env=SubprocVecEnv([partial(make,i) for i in range(8)],start_method='spawn')
 try:
  if STATE.exists() and CURRENT.exists():
   state=json.loads(STATE.read_text(encoding='utf8'));model=PPO.load(str(CURRENT),env=env,device='cpu',learning_rate=.0001)
  else:
   state=dict(weight=.5,lift_gain=.25,best_score=None,stale=0,iteration=0,total_env_steps=0,model='fifth-axis kd=.01; unvalidated')
   model=PPO('MlpPolicy',env,device='cpu',n_steps=256,batch_size=128,n_epochs=4,learning_rate=.0001,policy_kwargs=dict(log_std_init=-1.4),verbose=0,seed=101)
  configure(env,state['weight'],state['lift_gain']);save_json(STATE,state);print('STABILIZED CPU BASELINE '+json.dumps(state),flush=True)
  while True:
   start=time.monotonic();model.learn(total_timesteps=32768,reset_num_timesteps=False)
   assisted=evaluate(model,state['weight']);score=grade(assisted);passed=gate(assisted)
   state['iteration']+=1;state['total_env_steps']=int(model.num_timesteps)
   best=state['best_score'] is None or score>state['best_score']+.01
   if best:save_model(model,BEST);state['best_score']=score;state['stale']=0
   else:state['stale']+=1
   unassisted=evaluate(model,0.) if state['iteration']%2==0 or passed else None
   old=state['weight']
   if passed and state['weight']>0:
    save_model(model,ROOT/f"stabilized_pass_{int(round(state['weight']*100)):02d}.zip")
    state['weight']=round(max(0.,state['weight']-.05),2);state['best_score']=None;state['stale']=0
   elif state['stale']>=3:
    if BEST.exists():
     count=model.num_timesteps;model=PPO.load(str(BEST),env=env,device='cpu',learning_rate=.0001);model.num_timesteps=count;model.verbose=0
    gains=(.25,.4,.6);state['lift_gain']=gains[(gains.index(state['lift_gain'])+1)%len(gains)];state['stale']=0
   configure(env,state['weight'],state['lift_gain']);save_model(model,CURRENT);save_json(STATE,state)
   row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),iteration=state['iteration'],total_env_steps=state['total_env_steps'],
    elapsed_s=round(time.monotonic()-start,2),evaluated_weight=old,next_weight=state['weight'],score=score,best=best,
    passed_gate=passed,lift_gain=state['lift_gain'],assisted=assisted,unassisted=unassisted)
   with (ROOT/'stabilized_audits.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
   print('ITERATION '+json.dumps(row),flush=True)
 finally:env.close()
