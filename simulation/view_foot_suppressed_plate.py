"""Live, simulation-only view of the foot plate-suppression standing candidate."""
from pathlib import Path
import time
import mujoco
import mujoco.viewer
from guarded_pruned_servo import GuardedPrunedServo, HipRangeExceeded

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_foot_suppressed_plate_trial.xml'))
m.opt.timestep=0.00025
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0)
mujoco.mj_forward(m,d)
servo=GuardedPrunedServo(m,d)
print('Live simulation only: foot plate-suppression trial; no training. Close the viewer to stop.',flush=True)
print('Hip friction is idealized; collision coverage and hardware parameters are incomplete.',flush=True)
with mujoco.viewer.launch_passive(m,d) as viewer:
 viewer.cam.lookat[:]=[.04,-.04,.20]
 viewer.cam.distance=.9
 viewer.cam.azimuth=130
 viewer.cam.elevation=-15
 steps=max(1,round(.01/m.opt.timestep))
 while viewer.is_running():
  started=time.perf_counter()
  try:
   for _ in range(steps): servo.step()
  except HipRangeExceeded as exc:
   print(f'Simulation stopped: {exc}',flush=True)
   break
  viewer.sync()
  remaining=steps*m.opt.timestep-(time.perf_counter()-started)
  if remaining>0:time.sleep(remaining)
