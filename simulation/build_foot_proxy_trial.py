"""Try aligned box foot contact in an independent diagnostic model, no training."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
src=p/'diagnostic_range_collision_trial.xml';tree=ET.parse(src);root=tree.getroot()
m=mujoco.MjModel.from_xml_path(str(src));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
alignment=json.loads((p/'foot_collision_alignment_audit.json').read_text(encoding='utf8'))['results']
for row in alignment:
 leg=row['leg'];name=f'L{leg}_foot_proxy'
 geom=next(x for x in root.iter('geom') if x.get('name')==name)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part6')
 old=np.fromstring(geom.get('pos'),sep=' ')
 # Shift the box in parent-local coordinates so its initial lowest world point
 # matches the raw source mesh's lowest point, without altering the CAD mesh.
 delta_world=np.array([0.,0.,-row['proxy_minus_mesh_bottom_mm']*.001])
 new=old+d.xmat[body].reshape(3,3).T@delta_world
 geom.set('pos',' '.join(f'{v:.12g}' for v in new))
 pair=next(x for x in root.find('contact').findall('pair') if x.get('name')==f'floor_L{leg}_6')
 pair.set('geom2',name)
root.set('model','diagnostic_aligned_foot_proxy_trial')
out=p/'diagnostic_aligned_foot_proxy_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
rows=[]
for filename in (src,out):
 model=mujoco.MjModel.from_xml_path(str(filename));data=mujoco.MjData(model)
 mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
 servo=CandidateStandingServo(model,data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
    kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
 b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot_free');start=data.xipos[b].copy();max_tilt=0.;max_speed=0.;valid=True
 for step in range(round(4/model.opt.timestep)):
  old=float(data.time);servo.step()
  if not np.isfinite(data.qpos).all() or data.time<old+model.opt.timestep*.5:valid=False;break
  if step%20==0:
   max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(data.xmat[b].reshape(3,3)[2,2],-1,1)))))
   max_speed=max(max_speed,float(np.max(np.abs(data.qvel))))
 mujoco.mj_forward(model,data)
 row={'model':filename.name,'valid':valid,'duration_s':float(data.time),
      'height_change_mm':float((data.xipos[b,2]-start[2])*1000),'max_sampled_tilt_deg':max_tilt,
      'max_sampled_qvel':max_speed}
 rows.append(row);print(row,flush=True)
(p/'aligned_foot_proxy_trial_audit.json').write_text(json.dumps({'results':rows,'limitations':
 'Box bottom is aligned only at initial pose. Box support region is not validated against curved source mesh; dynamics are hypothetical. This trial is not for training.'},indent=2),encoding='utf8')
