"""Compare leg motion with a fixed body and no gravity or ground; not a walking test."""
from pathlib import Path
from dataclasses import replace
import tempfile,xml.etree.ElementTree as ET,json
import numpy as np
import mujoco
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec

p=Path(__file__).parent;tree=ET.parse(p/'diagnostic_sector_implicit_stiff_trial.xml');root=tree.getroot();body=root.find(".//body[@name='robot_free']");free=body.find("freejoint[@name='floating_base']")
key=root.find(".//key[@name='centered_start']");q=key.get('qpos').split();body.set('pos',' '.join(q[:3]));body.set('quat',' '.join(q[3:7]));body.remove(free);key.set('qpos',' '.join(q[7:]));root.find('option').set('gravity','0 0 0')
contact=root.find('contact')
for pair in list(contact):
 if pair.get('geom1')=='ground' or pair.get('geom2')=='ground':contact.remove(pair)
# Load from the model directory so existing relative mesh paths remain valid.
with tempfile.NamedTemporaryFile(suffix='.xml',dir=p,delete=False) as tmp:
 temp=Path(tmp.name);tree.write(tmp,encoding='utf-8',xml_declaration=True)
try:m=mujoco.MjModel.from_xml_path(str(temp))
finally:temp.unlink(missing_ok=True)
soles=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{i}_sole_trial_geom') for i in (0,1)];verts=[]
for geom in soles:
 mesh=int(m.geom_dataid[geom]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh]);verts.append(m.mesh_vert[start:start+count].copy())
rows=[]
for actuator in range(4):
 for sign in (-1,1):
  d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
  def soles_z():
   return np.asarray([np.min(d.geom_xpos[g,2]+verts[i]@d.geom_xmat[g].reshape(3,3)[2,:]) for i,g in enumerate(soles)])
  initial=soles_z();peak=np.zeros(2);low=np.zeros(2);pulse=np.full(5,1500.);pulse[actuator]+=sign*200.;s.set_pulse_us(pulse);stop=None
  for step in range(round(.8/m.opt.timestep)):
   before=d.time;s.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+.5*m.opt.timestep:stop='nonfinite_or_reset';break
   if step%80==0:
    mujoco.mj_forward(m,d);delta=soles_z()-initial;peak=np.maximum(peak,delta);low=np.minimum(low,delta)
  rows.append({'actuator':actuator,'sign':sign,'peak_sole_rise_mm':(peak*1000).tolist(),'peak_sole_drop_mm':(low*1000).tolist(),'stop_reason':stop})
  print(json.dumps(rows[-1]),flush=True)
out={'results':rows,'limitations':'Body fixed to initial pose, no gravity, ground contact removed, one actuator at a time for 0.8 s; checks numerical kinematic reachability only, not balance, real joint range, collision safety or hardware.'}
(p/'fixed_body_leg_motion_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
