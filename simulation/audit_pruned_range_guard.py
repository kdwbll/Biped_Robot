"""Test a simulation range guard without training or hardware control."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
from guarded_pruned_servo import GuardedPrunedServo, HipRangeExceeded

p=Path(__file__).parent
finer='--finer' in sys.argv
filename='diagnostic_finer_collision_trial.xml' if finer else 'diagnostic_pruned_nomerge_trial.xml'
m=mujoco.MjModel.from_xml_path(str(p/filename))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=GuardedPrunedServo(m,d);initial=d.qpos.copy();rows=[]
for leg,adr in enumerate(servo.hip_qadr):
 for angle in (-4.6,-4.4,4.4,4.6):
  d.qpos[:]=initial;d.qpos[adr]+=np.deg2rad(angle)
  try:servo.check_range();stopped=False
  except HipRangeExceeded:stopped=True
  expected=abs(angle)>4.5
  if stopped!=expected:raise RuntimeError((leg,angle,stopped))
  rows.append({'leg':leg,'injected_delta_deg':angle,'expected_stop':expected,'actual_stop':stopped})
d.qpos[:]=initial;mujoco.mj_forward(m,d)
d.qpos[servo.hip_qadr[0]]+=np.deg2rad(4.6);time_before=float(d.time)
try:servo.step();raise RuntimeError('Out-of-range step was accepted')
except HipRangeExceeded:pass
if d.time!=time_before:raise RuntimeError('Guard advanced the out-of-range simulation')
d.qpos[:]=initial;mujoco.mj_forward(m,d)
d.qvel[0]=np.nan
try:servo.step();raise RuntimeError('Nonfinite state was accepted')
except HipRangeExceeded:pass
d.qvel[0]=0.;mujoco.mj_forward(m,d)
other=mujoco.MjModel.from_xml_path(str(p/'diagnostic_range_collision_trial.xml'))
other_data=mujoco.MjData(other);mujoco.mj_resetDataKeyframe(other,other_data,0)
try:GuardedPrunedServo(other,other_data);raise RuntimeError('Wrong model accepted')
except ValueError:pass
d.qpos[servo.hip_qadr[0]]+=np.deg2rad(.1)
try:GuardedPrunedServo(m,d);raise RuntimeError('Non-keyframe hip pose accepted')
except ValueError:pass
d.qpos[:]=initial;mujoco.mj_forward(m,d)
for step in range(round(4/m.opt.timestep)):
 if d.time>=2:servo.set_pulse_us([1500,1500,1500,1500,1510])
 servo.step()
out={'model':filename,'boundary_cases':rows,'out_of_range_step_blocked_before_advancing':True,'nonfinite_state_rejected':True,'wrong_model_rejected':True,'non_keyframe_pose_rejected':True,'guarded_run_seconds':float(d.time),
 'guard_threshold_deg':4.5,'assumed_collision_pruning_range_deg':5.,
 'limitations':'Guard exists only when this Python class runs. Direct XML use, other controllers, within-step overshoot, excluded collision pairs and model uncertainty remain.'}
(p/('finer_range_guard_audit.json' if finer else 'pruned_range_guard_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print('guard boundaries checked:',len(rows),'guarded small-command run seconds:',d.time,flush=True)
