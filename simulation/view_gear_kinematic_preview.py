"""Illustrate gear/leg directions by kinematics only: no dynamics or training."""
from pathlib import Path
import math
import time
import mujoco
import mujoco.viewer

p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
names=['hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate']
ids=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]
initial=d.qpos.copy();started=time.monotonic()
print('Kinematic visualization only: no ground dynamics, torque simulation or training.',flush=True)
print('Pinion oscillates +/- 8 degrees; large gears and attached legs rotate in opposite directions.',flush=True)
with mujoco.viewer.launch_passive(m,d) as viewer:
 viewer.cam.lookat[:]=[.04,-.04,.15];viewer.cam.distance=.65;viewer.cam.azimuth=130;viewer.cam.elevation=-5
 while viewer.is_running():
  delta=math.radians(8)*math.sin(2*math.pi*(time.monotonic()-started)/4)
  d.qpos[:]=initial
  d.qpos[ids[0]]+=delta;d.qpos[ids[1]]-=.5*delta;d.qpos[ids[2]]-=.5*delta
  mujoco.mj_forward(m,d);viewer.sync();time.sleep(1/60)
