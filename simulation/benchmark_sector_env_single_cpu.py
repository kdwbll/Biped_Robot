"""Benchmark a single simulation-only environment without optimization/training."""
from pathlib import Path
import json,time
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;t=time.perf_counter();env=SectorStandingEnv(episode_seconds=1,model_variant='implicit_stiff',friction_range=(.05,.2),max_gear_residual_deg=.2,max_sampled_self_contact_n=20.,max_pwm_delta_us_per_control=1.);env.reset(seed=1);setup=time.perf_counter()-t
start=time.perf_counter()
for step in range(50):
 _,_,terminated,truncated,info=env.step(np.zeros(5))
 if terminated or truncated:break
elapsed=time.perf_counter()-start
out={'control_steps':step+1,'simulated_seconds':float(info['simulation_time']),'setup_seconds':setup,'wall_seconds':elapsed,'control_steps_per_second':(step+1)/elapsed,'simulation_realtime_factor':info['simulation_time']/elapsed,'terminated':terminated,'truncated':truncated,'reason':info['reason'],'physics_backend':'MuJoCo CPU','limitations':'One local single-process run, no renderer, no PPO or GPU physics. Host load and hardware affect timing; not parallel throughput.'}
(p/'sector_env_single_cpu_benchmark.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
