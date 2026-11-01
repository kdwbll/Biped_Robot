import numpy as np
from biped_env import BipedEnv
from stable_baselines3 import PPO
policy=PPO.load(r'C:\Practice_project\biped_sim\ppo_experiment_4096.zip',device='cpu')
for file in ['biped_filtered_collision_experimental.xml','biped_mirrored_missing_mass_experiment.xml']:
 for mode in ('neutral','ppo'):
  e=BipedEnv(episode_seconds=4,model_path=file);o,_=e.reset(seed=42);peak=0.;roll0=0
  for k in range(200):
   a=np.zeros(5) if mode=='neutral' else policy.predict(o,deterministic=True)[0];o,r,term,trunc,info=e.step(a);R=e.data.xmat[e.body].reshape(3,3);roll=np.degrees(np.arctan2(R[2,1],R[2,2]));peak=max(peak,abs(roll))
   if k==0:roll0=roll
   if term or trunc:break
  print(file,mode,'sec',round((k+1)*.02,2),'initial roll',round(roll0,3),'max |roll|',round(peak,3),'final roll',round(roll,3),'fallen',term)
