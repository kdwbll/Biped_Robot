"""Compare plane distances for original compound hull vs separate sole/upper hulls."""
from pathlib import Path
import json
import mujoco
import numpy as np
p=Path(__file__).parent
models=[mujoco.MjModel.from_xml_path(str(p/f)) for f in ('diagnostic_foot_suppressed_plate_trial.xml','diagnostic_separate_sole_contact_trial.xml')]
rows=[]
for axis in ('x','y'):
 for deg in (-15,-5,0,5,15):
  states=[]
  for m in models:
   d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
   root=next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE);q=m.jnt_qposadr[root];ang=np.deg2rad(deg)/2;quat=np.array([np.cos(ang),np.sin(ang) if axis=='x' else 0,np.sin(ang) if axis=='y' else 0,0])
   old=d.qpos[q+3:q+7].copy();result=np.zeros(4);mujoco.mju_mulQuat(result,quat,old);d.qpos[q+3:q+7]=result;mujoco.mj_forward(m,d);ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');ls=[]
   for leg in (0,1):
    names=[f'L{leg}_visual6'] if len(states)==0 else [f'L{leg}_sole_trial_geom',f'L{leg}_foot_upper_trial_geom']
    vals=[]
    for name in names:
     g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
     vals.append(float(mujoco.mj_geomDistance(m,d,ground,g,1,None)*1000))
    ls.append(vals)
   states.append(ls)
  for leg in (0,1):rows.append({'axis':axis,'deg':deg,'leg':leg,'old_combined_hull_mm':states[0][leg][0],'new_sole_mm':states[1][leg][0],'new_upper_mm':states[1][leg][1],'minimum_difference_mm':min(states[1][leg])-states[0][leg][0]})
out={'results':rows,'max_abs_difference_mm':max(abs(x['minimum_difference_mm']) for x in rows),'limitations':'Geometric free-base tilts of keyframe, not necessarily reachable or physically stable. Compare minimum plane distance only; not full contact manifold, friction or all poses.'}
(p/'separate_sole_ground_distance_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('max |difference| mm',out['max_abs_difference_mm'])
