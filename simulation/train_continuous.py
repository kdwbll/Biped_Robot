"""Continuous simulation-only PPO with stage gates, rollback, and durable state.

Run with 8 CPU environments. Stop via stop_continuous_training.ps1; each
iteration is checkpointed. The simulation includes virtual balance and weight
support, which are explicitly removed in every second audit.
"""
import json,os,time,traceback
from functools import partial
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv
from train_fast_curriculum import evaluate
from live_telemetry import open_buffer,write_slot

ROOT=Path(__file__).parent
STATE=ROOT/'continuous_state.json'
BEST=ROOT/'continuous_best.zip'
CURRENT=ROOT/'continuous_current.zip'
ITERATION=32768
EPISODE_SECONDS=12.

class AdaptiveEnv(ResidualGaitEnv):
 def __init__(self,slot):
  super().__init__(episode_seconds=EPISODE_SECONDS,support_gain=1.3125,weight_support=.35);self.lift_gain=.25
  self.telemetry_map,self.telemetry=open_buffer();self.slot=slot;self.last_snapshot=0.
 def set_lift_gain(self,value):self.lift_gain=float(value)
 def step(self,action):
  obs,reward,term,trunc,info=super().step(action);leg=info.get('phase_swing',0)
  if info.get('foot_contacts',[False,False])[1-leg]:
   reward+=(self.lift_gain if leg==0 else .10)*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
  now=time.monotonic()
  if now-self.last_snapshot>.15:
   write_slot(self.telemetry,self.slot,self.data,self.weight_support);self.last_snapshot=now
  return obs,reward,term,trunc,info
def make(slot):return AdaptiveEnv(slot)

def save_json(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');os.replace(temp,path)
def save_model(model,path):
 temp=path.with_name(path.stem+'.tmp.zip');model.save(str(temp));os.replace(temp,path)
def grade(rows):
 return float(np.mean([(0 if r['fell'] else 10)+min(r['alternations'],6)*2
   +sum(min(max(v,0)/.003,1) for v in r['peak_clearance_m'])
   +float(np.clip(r['forward_m']/.03,-1,1)) for r in rows]))
def gate(rows):return all(not r['fell'] and r['forward_m']>0 and r['alternations']>=3 for r in rows)
def configure(env,weight,gain):
 env.env_method('set_weight_support',weight);env.env_method('set_support',3.75*weight);env.env_method('set_lift_gain',gain)

if __name__=='__main__':
 torch.set_num_threads(1);telemetry_map,telemetry=open_buffer();telemetry[:]=0.
 env=SubprocVecEnv([partial(make,i) for i in range(8)],start_method='spawn')
 try:
  state=json.loads(STATE.read_text(encoding='utf8')) if STATE.exists() else None
  if state and CURRENT.exists():
   model=PPO.load(str(CURRENT),env=env,device='cpu',learning_rate=.0001)
  else:
   model=PPO.load(str(ROOT/'fast_cpu_294912.zip'),env=env,device='cpu',learning_rate=.0001)
   state=dict(weight=.35,lift_gain=.25,best_score=None,stale=0,iteration=0,total_env_steps=294912)
  model.verbose=0
  configure(env,state['weight'],state['lift_gain'])
  print('CONTINUOUS TRAINING START',json.dumps(state),flush=True)
  while True:
   started=time.monotonic();model.learn(total_timesteps=ITERATION,reset_num_timesteps=False,progress_bar=False)
   assisted=evaluate(model,state['weight']);score=grade(assisted);passed=gate(assisted)
   state['iteration']+=1;state['total_env_steps']=int(model.num_timesteps)
   is_best=state['best_score'] is None or score>state['best_score']+.01
   if is_best:
    save_model(model,BEST);state['best_score']=score;state['stale']=0
   else:state['stale']+=1
   unassisted=evaluate(model,0.) if state['iteration']%2==0 or passed else None
   longer=evaluate(model,state['weight'],episode_seconds=12.,seeds=(9201,)) if state['iteration']%2==0 else None
   previous_weight=state['weight']
   if passed and state['weight']>0:
    save_model(model,ROOT/f"continuous_pass_{int(round(state['weight']*100)):02d}.zip")
    state['weight']=round(max(0.,state['weight']-.05),2);state['best_score']=None;state['stale']=0
   elif state['stale']>=3:
    if BEST.exists():
     steps=model.num_timesteps;model=PPO.load(str(BEST),env=env,device='cpu',learning_rate=.0001);model.num_timesteps=steps;model.verbose=0
    gains=(.25,.4,.6);state['lift_gain']=gains[(gains.index(state['lift_gain'])+1)%len(gains)] if state['lift_gain'] in gains else .25
    state['stale']=0
   configure(env,state['weight'],state['lift_gain']);save_model(model,CURRENT);save_json(STATE,state)
   record=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),iteration=state['iteration'],total_env_steps=state['total_env_steps'],
     elapsed_s=round(time.monotonic()-started,2),evaluated_weight=previous_weight,next_weight=state['weight'],
     score=score,best=is_best,passed_gate=passed,lift_gain=state['lift_gain'],assisted=assisted,unassisted=unassisted,longer_test=longer)
   with (ROOT/'continuous_audits.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
   print('ITERATION '+json.dumps(record,ensure_ascii=False),flush=True)
 finally:env.close()
