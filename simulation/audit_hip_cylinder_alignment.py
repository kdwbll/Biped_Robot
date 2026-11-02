"""Compare CAD cylinder axes to the simulated hip hinge axes."""
from pathlib import Path
import json
import mujoco
import numpy as np

p=Path(__file__).parent
cad=json.loads((p/'cad_hip_cylinders_audit.json').read_text(encoding='utf8'))
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');R=d.xmat[root].reshape(3,3);origin=d.xpos[root]
rows=[]
for leg in (0,1):
 target_y=.032 if leg==0 else -.073
 def match(component,volume,radius):
  candidates=[x for x in cad if x['component']==component and abs(x['body_volume_mm3']-volume)<200 and
    abs(x['radius_mm']-radius)<.01 and abs(x['axis_point_m'][0]-.042)<.0001 and abs(x['axis_point_m'][1]-target_y)<.0001]
  if len(candidates)!=1:raise RuntimeError((component,volume,radius,len(candidates)))
  return candidates[0]
 hole=match('frame',143769,20.);pillar=match(f'leg{leg}',129952,19.75)
 joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate')
 hinge=R.T@(d.xanchor[joint]-origin);axis=R.T@d.xaxis[joint]
 cad_axis=np.asarray(pillar['axis_direction']);cad_axis/=np.linalg.norm(cad_axis)
 cad_point=np.asarray(pillar['axis_point_m']);difference=hinge-cad_point
 offset=np.linalg.norm(difference-cad_axis*np.dot(difference,cad_axis))*1000
 align=abs(float(np.dot(axis/np.linalg.norm(axis),cad_axis)))
 rows.append({'leg':leg,'frame_hole_radius_mm':hole['radius_mm'],'pillar_radius_mm':pillar['radius_mm'],
  'nominal_radial_clearance_mm':hole['radius_mm']-pillar['radius_mm'],
  'cad_axis_xy_mm':(cad_point[:2]*1000).tolist(),'sim_axis_xy_mm':(hinge[:2]*1000).tolist(),
  'axis_line_offset_mm':float(offset),'axis_direction_abs_dot':align})
out={'results':rows,'limitations':'Nominal CAD geometry and model keyframe, not manufactured clearance or proof of all-axis motion. User confirmed pillar and gear are rigidly linked.'}
if any(x['axis_line_offset_mm']>.01 or x['axis_direction_abs_dot']<.9999 for x in rows):raise RuntimeError(out)
(p/'hip_cylinder_alignment_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
