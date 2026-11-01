import time,sys,numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from randomized_biped_env import PerturbedBipedEnv

def make(): return PerturbedBipedEnv(episode_seconds=4)
if __name__=='__main__':
 n=int(sys.argv[1]); v=DummyVecEnv([make]) if n==1 else SubprocVecEnv([make for _ in range(n)],start_method='spawn');v.reset();start=time.perf_counter()
 for _ in range(100):v.step(np.zeros((n,5),dtype=np.float32))
 print('workers',n,'samples/sec',round(n*100/(time.perf_counter()-start),1),flush=True);v.close()
