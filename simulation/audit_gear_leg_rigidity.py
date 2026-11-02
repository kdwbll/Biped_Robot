"""Check that a hip joint rotates its whole leg and the pinion follows its shaft."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np

p=Path(__file__).parent
smooth='--smooth-hip' in sys.argv
phase25='--25-tooth' in sys.argv
m=mujoco.MjModel.from_xml_path(str(p/('diagnostic_smooth_hip_trial.xml' if smooth else 'diagnostic_post_shaft_fix_gear_phase_25.xml' if phase25 else 'diagnostic_post_shaft_fix_gear_phase.xml')));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
pinion=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'phase_pinion_visual')
shaft_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'hypothetical_motor_shaft')
def relative_legs(mount,bodies):
 R=d.xmat[mount].reshape(3,3);x=d.xpos[mount]
 return [(R.T@(d.xpos[b]-x),R.T@d.xmat[b].reshape(3,3)) for b in bodies]
def pinion_relative():
 R=d.xmat[shaft_body].reshape(3,3)
 return R.T@(d.geom_xpos[pinion]-d.xpos[shaft_body]),R.T@d.geom_xmat[pinion].reshape(3,3)
legs=[]
for leg in (0,1):
 mount=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_mount')
 bodies=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part{i}') for i in range(7)]
 hip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate')
 before=relative_legs(mount,bodies);d.qpos[m.jnt_qposadr[hip]]+=.01;mujoco.mj_forward(m,d)
 after=relative_legs(mount,bodies)
 legs.append({'leg':leg,'max_relative_position_change_m':max(float(np.linalg.norm(a[0]-b[0])) for a,b in zip(before,after)),
              'max_relative_rotation_matrix_change':max(float(np.linalg.norm(a[1]-b[1])) for a,b in zip(before,after))})
 d.qpos[m.jnt_qposadr[hip]]-=.01;mujoco.mj_forward(m,d)
pin_before=pinion_relative();d.qpos[m.jnt_qposadr[shaft]]+=.01;mujoco.mj_forward(m,d);pin_after=pinion_relative()
position=max(x['max_relative_position_change_m'] for x in legs)
rotation=max(x['max_relative_rotation_matrix_change'] for x in legs)
pinpos=float(np.linalg.norm(pin_before[0]-pin_after[0]));pinrot=float(np.linalg.norm(pin_before[1]-pin_after[1]))
out={'hip_increment_rad':.01,'shaft_increment_rad':.01,
     'max_leg_body_relative_position_change_m':position,'max_leg_body_relative_rotation_matrix_change':rotation,
     'legs':legs,
     'pinion_to_shaft_relative_position_change_m':pinpos,'pinion_to_shaft_relative_rotation_matrix_change':pinrot,
     'note':'Kinematic parenting check only; does not prove physical mounting, collision or torque.'}
if max(position,rotation,pinpos,pinrot)>1e-8:raise RuntimeError(out)
(p/('smooth_hip_gear_leg_rigidity_audit.json' if smooth else 'gear_leg_rigidity_25_audit.json' if phase25 else 'gear_leg_rigidity_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print('All seven bodies per leg move rigidly with their mount under isolated hip perturbations; pinion visual follows shaft.')
print('max relative position/rotation errors',position,rotation,'pinion',pinpos,pinrot)
