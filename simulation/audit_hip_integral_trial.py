"""Compare central servo integral gains in small bidirectional ground tests."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
single='--positive-ki10' in sys.argv
ablate='--ablate-peak-pair' in sys.argv
chassis0='--chassis0' in sys.argv
finer='--finer' in sys.argv
model_path=p/('diagnostic_chassis0_collision_trial.xml' if chassis0 else 'diagnostic_finer_collision_trial.xml' if finer else 'diagnostic_post_shaft_fix_gear_phase_25.xml')
if ablate:
 tree=ET.parse(model_path);contact=tree.getroot().find('contact');removed=0
 for pair in list(contact.findall('pair')):
  if {pair.get('geom1'),pair.get('geom2')}=={'dc_chassis_0_7','tower_L1_tower_8'}:
   contact.remove(pair);removed+=1
 if removed!=1:raise RuntimeError(f'Expected one peak pair; found {removed}')
 tree.getroot().set('model','diagnostic_peak_pair_ablation')
 model_path=p/'diagnostic_peak_pair_ablation.xml';tree.write(model_path,encoding='utf8',xml_declaration=True)
for ki in ((10.,) if single else (6.,10.,15.)):
 for delta in ((10,) if single else (-10,10)):
  m=mujoco.MjModel.from_xml_path(str(model_path))
  d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
      kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,ki])
  ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');initial=servo.initial_positions[4]
  peak=0.;peak_pair=None;peak_time=None;valid=True
  for step in range(round(6/m.opt.timestep)):
   if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1500+delta])
   old=d.time;servo.step()
   if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:valid=False;break
   if step%20==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
     if f[0]>peak:
      peak=float(f[0]);peak_time=float(d.time)
      peak_pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  mujoco.mj_forward(m,d)
  row={'ki':ki,'pulse_delta_us':delta,'valid':valid,'actual_deg':float(np.degrees(d.actuator_length[4]-initial)),
       'error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])), 'sampled_peak_self_contact_n':peak,
       'sampled_peak_pair':peak_pair,'sampled_peak_time_s':peak_time}
  rows.append(row);print(row,flush=True)
(p/('chassis0_hip_integral_positive_audit.json' if chassis0 and single else 'chassis0_hip_integral_audit.json' if chassis0 else 'finer_hip_integral_positive_audit.json' if finer and single else 'hip_peak_pair_ablation_audit.json' if ablate else 'hip_integral_positive_peak_audit.json' if single else 'hip_integral_trial_audit.json')).write_text(json.dumps({'results':rows,
  'limitations':'Hypothetical simulation, no physical validation and collision coverage is incomplete.'},indent=2),encoding='utf8')
