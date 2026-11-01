import numpy as np
from randomized_biped_env import PerturbedBipedEnv
x=PerturbedBipedEnv();a,info=x.reset(seed=13);b,info2=x.reset(seed=13);c,info3=x.reset(seed=14)
assert np.allclose(a,b) and not np.allclose(a,c)
print('same seed reproducible, different seed distinct:',info['initial_angular_velocity'],info3['initial_angular_velocity'])
