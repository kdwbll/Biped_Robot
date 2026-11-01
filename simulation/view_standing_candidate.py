"""Show the experimental simulated stance controller; no policy training."""
from pathlib import Path
from dataclasses import replace
import time
import mujoco
import mujoco.viewer
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
print('Simulation only: experimental stance controller and refined collision proxies.',flush=True)
print('Collision coverage and gear contact remain unverified; +/-5 deg hip range is an assumption.',flush=True)
print('Close viewer to stop.',flush=True)
with mujoco.viewer.launch_passive(m,d) as viewer:
 viewer.cam.lookat[:]=[.04,-.04,.20]
 viewer.cam.distance=.9
 viewer.cam.azimuth=130
 viewer.cam.elevation=-15
 n=max(1,round(.01/m.opt.timestep))
 while viewer.is_running():
  started=time.perf_counter()
  for _ in range(n):servo.step()
  viewer.sync()
  remaining=n*m.opt.timestep-(time.perf_counter()-started)
  if remaining>0:time.sleep(remaining)
