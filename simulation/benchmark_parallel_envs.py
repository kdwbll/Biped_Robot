"""Compare physical simulation throughput with 4 and 8 independent CPU workers."""
import json,sys,time
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv

def make():return ResidualGaitEnv(support_gain=1.5,weight_support=.4)

if __name__=='__main__':
    n=int(sys.argv[1]);env=SubprocVecEnv([make for _ in range(n)],start_method='spawn')
    env.reset();action=np.zeros((n,5),dtype=np.float32)
    for _ in range(8):env.step(action)
    start=time.perf_counter()
    for _ in range(80):env.step(action)
    elapsed=time.perf_counter()-start;env.close()
    print(json.dumps(dict(workers=n,steps=n*80,seconds=elapsed,env_steps_per_second=n*80/elapsed)),flush=True)
