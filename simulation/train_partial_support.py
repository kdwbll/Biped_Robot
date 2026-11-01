"""Focus on foot clearance at 64% body support before fading further."""
import json,sys
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv
from train_residual_gait import audit
ROOT=Path(__file__).parent
MODE=sys.argv[1] if len(sys.argv)>1 else ''
NEXT_STAGE=MODE in ('half','half2','half3','half4','forty')
WEIGHT=.4 if MODE=='forty' else (.5 if NEXT_STAGE else .64)
PREFIX={'half':'partial50','half2':'partial50b','half3':'focusleft','half4':'drift50','forty':'partial40'}.get(MODE,'partial')
START={'half':49152,'half2':81920,'half3':98304,'half4':114688,'forty':131072}.get(MODE,32768)

class FocusedLiftEnv(ResidualGaitEnv):
    def __init__(self):super().__init__(support_gain=3.75*WEIGHT,weight_support=WEIGHT)
    def step(self,action):
        obs,reward,term,trunc,info=super().step(action)
        leg=info.get('phase_swing',0);other=1-leg
        if info.get('foot_contacts',[False,False])[other]:
            gain=.25 if MODE in ('half3','half4','forty') and leg==0 else .10
            reward+=gain*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
        if MODE=='half4':reward-=2.0*float(info.get('support_drift_m',0.))
        return obs,reward,term,trunc,info

def make():return FocusedLiftEnv()

class Audit(BaseCallback):
    def __init__(self):super().__init__();self.next_eval=START+16384
    def _on_step(self):
        if self.num_timesteps>=self.next_eval:
            rows=audit(self.model);result=dict(timesteps=self.num_timesteps,training_weight_support=WEIGHT,
                    falls=sum(r['fell'] for r in rows),episodes=rows)
            (ROOT/f'{PREFIX}_gpu_{self.num_timesteps}_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            print('UNASSISTED AUDIT '+json.dumps(result),flush=True);self.next_eval+=16384
        return True

if __name__=='__main__':
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);env=SubprocVecEnv([make for _ in range(4)],start_method='spawn')
    source={'half':ROOT/'training_checkpoints/partial_gpu_49152_steps.zip',
            'half2':ROOT/'partial50_gpu_81920.zip',
            'half3':ROOT/'training_checkpoints/partial50b_gpu_98304_steps.zip',
            'half4':ROOT/'training_checkpoints/focusleft_gpu_114688_steps.zip',
            'forty':ROOT/'focusleft_gpu_131072.zip'}.get(MODE,ROOT/'training_checkpoints/residual_gpu_32768_steps.zip')
    model=PPO.load(str(source),env=env,device='cuda',learning_rate=.0001)
    ckpt=CheckpointCallback(save_freq=2048,save_path=str(ROOT/'training_checkpoints'),name_prefix=f'{PREFIX}_gpu')
    model.learn(total_timesteps=32768,reset_num_timesteps=False,callback=CallbackList([Audit(),ckpt]))
    model.save(str(ROOT/f'{PREFIX}_gpu_{START+32768}.zip'));env.close();print('partial support training complete',flush=True)
