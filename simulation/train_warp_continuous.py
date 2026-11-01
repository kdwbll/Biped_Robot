"""Continuous experimental GPU PPO; CPU cross-checks gate support reductions."""
import json,time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from warp_vec_env import WarpVecEnv
from train_stabilized_cpu import evaluate
from train_continuous import grade,gate,save_model,save_json
ROOT=Path(__file__).parent
STATE=ROOT/'warp_training_state.json';CURRENT=ROOT/'warp_training_current.zip';BEST=ROOT/'warp_training_best.zip'
STEPS=32768
if __name__=='__main__':
 torch.set_num_threads(1)
 state=json.loads(STATE.read_text(encoding='utf8')) if STATE.exists() and CURRENT.exists() else None
 if state is None:state=dict(weight=.5,iteration=0,total_env_steps=32768,best_score=None,stale=0,model='experimental GPU stabilized fifth axis')
 state.setdefault('lift_gain',.25)
 env=WarpVecEnv(nworld=32,weight=state['weight'],seconds=12.,seed=318,telemetry_tag='Local\\BipedRobotGpuTelemetryV1')
 env.lift_gain=state['lift_gain']
 source=CURRENT if CURRENT.exists() and STATE.exists() else ROOT/'warp_trial_32768.zip'
 model=PPO.load(str(source),env=env,device='cuda',learning_rate=.0001);model.verbose=0
 save_json(STATE,state);print('GPU EXPERIMENTAL TRAINING START '+json.dumps(state),flush=True)
 try:
  while True:
   start=time.monotonic();model.learn(total_timesteps=STEPS,reset_num_timesteps=False)
   elapsed=time.monotonic()-start;state['iteration']+=1;state['total_env_steps']=int(model.num_timesteps)
   assisted=unassisted=None;passed=False;score=None;old=state['weight']
   if state['iteration']%2==0:
    # CPU physics validation makes one prediction at a time; keeping the tiny
    # policy on the CPU avoids a device round trip on every validation step.
    model.policy.to('cpu')
    try:
     assisted=evaluate(model,state['weight']);unassisted=evaluate(model,0.)
    finally:model.policy.to('cuda')
    score=grade(assisted);passed=gate(assisted)
    if state['best_score'] is None or score>state['best_score']+.01:
     save_model(model,BEST);state['best_score']=score;state['stale']=0
    else:state['stale']+=1
    if passed and state['weight']>0:
     save_model(model,ROOT/f"warp_pass_{int(round(state['weight']*100)):02d}.zip")
     state['weight']=round(max(0.,state['weight']-.05),2);state['best_score']=None;state['stale']=0
    elif state['stale']>=3 and BEST.exists():
     count=model.num_timesteps;model=PPO.load(str(BEST),env=env,device='cuda',learning_rate=.0001);model.num_timesteps=count;model.verbose=0;state['stale']=0
     gains=(.25,.4,.6);state['lift_gain']=gains[(gains.index(state['lift_gain'])+1)%len(gains)]
   env.weight=state['weight'];env.lift_gain=state['lift_gain'];save_model(model,CURRENT);save_json(STATE,state)
   row=dict(iteration=state['iteration'],total_env_steps=state['total_env_steps'],train_seconds=elapsed,
    env_steps_per_second=STEPS/elapsed,evaluated_weight=old,next_weight=state['weight'],passed_gate=passed,
    score=score,lift_gain=state['lift_gain'],cpu_assisted=assisted,cpu_unassisted=unassisted)
   with (ROOT/'warp_training_audits.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
   print('GPU ITERATION '+json.dumps(row),flush=True)
 finally:env.close()
