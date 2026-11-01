import numpy as np
from warp_vec_env import WarpVecEnv
e=WarpVecEnv(nworld=2,seconds=.04)
o=e.reset();print('RESET',o.shape,np.isfinite(o).all(),flush=True)
for i in range(4):
 o,r,d,info=e.step(np.zeros((2,5)));print('STEP',i,o.shape,r,d,info[0]['overflow'],flush=True)
e.close()

# Verify one world's timeout does not reset the other world.
e=WarpVecEnv(nworld=2,seconds=1.,seed=7);e.reset()
e.count[0]=e.horizon-1
_,_,done,info=e.step(np.zeros((2,5)))
assert done.tolist()==[True,False],done
assert e.count.tolist()==[0,1],e.count
assert 'terminal_observation' in info[0] and 'terminal_observation' not in info[1]
print('INDEPENDENT_RESET',done.tolist(),e.count.tolist(),flush=True);e.close()
