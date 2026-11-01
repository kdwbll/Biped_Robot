"""Measure relative simulation-step cost of two local diagnostic models."""
from pathlib import Path
from dataclasses import replace
import json
import time
import mujoco
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
for name in ('diagnostic_range_collision_trial.xml','diagnostic_nomerge_collision_trial.xml','diagnostic_pruned_nomerge_trial.xml','diagnostic_finer_collision_trial.xml'):
 m=mujoco.MjModel.from_xml_path(str(p/name));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
    kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
 steps=round(.25/m.opt.timestep);start=time.perf_counter()
 for _ in range(steps):
  before=d.time;servo.step()
  if d.time<before+m.opt.timestep*.5:raise RuntimeError('Automatic reset')
 elapsed=time.perf_counter()-start
 rows.append({'model':name,'contact_pairs':m.npair,'simulated_seconds':float(d.time),'wall_seconds':elapsed,'steps_per_wall_second':steps/elapsed})
 print(rows[-1],flush=True)
(p/'nomerge_collision_benchmark.json').write_text(json.dumps({'results':rows,'limitations':'Single local short run; results depend on machine load and are not GPU or training benchmarks.'},indent=2),encoding='utf8')
