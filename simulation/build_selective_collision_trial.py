"""Controlled partial collision coverage using high-resolution convex proxies.

Initial overlapping/nearby pairs are deliberately excluded. This is a diagnostic,
not complete collision coverage, and must not be used for training.
"""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
src=p/'diagnostic_highres_gap_only.xml'
tree=ET.parse(src);root=tree.getroot();contact=root.find('contact')
m=mujoco.MjModel.from_xml_path(str(src));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
chassis=[(i,n) for i,n in enumerate(names) if n and n.startswith('highres_chassis5_')]
towers=[(i,n) for i,n in enumerate(names) if n and (n.startswith('highres_L0_') or n.startswith('highres_L1_'))]
if len(chassis)!=96 or len(towers)!=192:raise RuntimeError('Expected complete high-resolution proxy sets')
old_removed=0
for pair in list(contact.findall('pair')):
 a,b=pair.get('geom1'),pair.get('geom2')
 if (a.startswith('dc_chassis_5_') and (b.startswith('tower_') or b.startswith('refined_L'))) or (b.startswith('dc_chassis_5_') and (a.startswith('tower_') or a.startswith('refined_L'))):
  contact.remove(pair);old_removed+=1
skipped=[];added=0
for ai,a in chassis:
 for bi,b in towers:
  dist=float(mujoco.mj_geomDistance(m,d,ai,bi,.001,None))
  if dist<=.0003:
   skipped.append({'geoms':[a,b],'initial_signed_distance_mm':dist*1000});continue
  ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001');added+=1
root.set('model','diagnostic_selective_collision_trial')
path=p/'diagnostic_selective_collision_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
out={'margin_mm':.3,'old_proxy_pairs_removed':old_removed,'new_pair_count':added,'excluded_at_initial_pose':skipped,
 'limitations':'Pairs within 0.3 mm at initial pose are excluded for false overlap; coverage is incomplete throughout motion. CAD bodies are not convex proxy shapes. Do not use for training.'}
(p/'selective_collision_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('old pairs removed',old_removed,'new pairs',added,'excluded near initial',len(skipped),flush=True)
rows=[]
for xml in (p/'diagnostic_post_shaft_fix_gear_phase_25.xml',path):
 model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
 mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
 servo=CandidateStandingServo(model,data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
  kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
 ground=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ground');peak=0.;valid=True;peakpair=None
 for step in range(round(.5/model.opt.timestep)):
  old=data.time;servo.step()
  if not np.isfinite(data.qpos).all() or data.time<old+model.opt.timestep*.5:valid=False;break
  if step%10==0:
   for i,c in enumerate(data.contact[:data.ncon]):
    if ground in (c.geom1,c.geom2):continue
    f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f)
    if f[0]>peak:
     peak=float(f[0]);peakpair=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
 row={'model':xml.name,'valid':valid,'duration_s':float(data.time),'sampled_peak_self_contact_n':peak,'peak_pair':peakpair}
 rows.append(row);print(row,flush=True)
(p/'selective_collision_trial_audit.json').write_text(json.dumps({'results':rows,'limitations':out['limitations']},indent=2),encoding='utf8')
