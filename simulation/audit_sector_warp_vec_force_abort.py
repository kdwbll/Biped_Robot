"""Exercise actual GPU VecEnv contact-force abort with a tiny test threshold."""
from pathlib import Path
import json,sys
import numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,contact_limit_n=.0001)
try:
 env.reset();reason=None
 try:env.step(np.zeros((4,5),dtype=np.float32))
 except RuntimeError as e:reason=str(e)
 flags=env.flags.numpy().tolist();times=env.gd.time.numpy().tolist()
 passed=reason is not None and 'GPU diagnostic batch aborted' in reason and any(x&8 for x in flags)
 out={'contact_limit_n':.0001,'abort_triggered':passed,'flags':flags,'times_s':times,'reason':reason,'limitations':'Very small artificial threshold is only to exercise a live path; not a physical safety threshold. Four-world short test; no policy or training.'}
 (p/'sector_warp_vec_force_abort_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
 if not passed:raise RuntimeError('Live contact abort was not observed')
finally:env.close()
