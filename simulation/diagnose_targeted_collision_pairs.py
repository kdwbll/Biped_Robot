"""Ablate high-force pairs for diagnosis only; does not validate physical collision."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
contacts=json.loads((p/'standing_contact_forces_audit.json').read_text(encoding='utf8'))
suspects={tuple(sorted(r['geoms'])) for r in contacts if 'ground' not in r['geoms'] and r['normal_force_n']>100.}
tree=ET.parse(p/'biped_standing_gear_candidate.xml');root=tree.getroot();group=root.find('contact');removed=[]
for pair in list(group.findall('pair')):
 key=tuple(sorted((pair.get('geom1'),pair.get('geom2'))))
 if key in suspects:group.remove(pair);removed.append(key)
if set(removed)!=suspects:raise RuntimeError('Not all suspect pairs were present')
root.set('model','diagnostic_high_force_pairs_removed');ET.indent(root)
out=p/'diagnostic_high_force_pairs_removed.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');z0=d.xipos[body,2]
for _ in range(round(6/m.opt.timestep)):
 s.step()
 if not np.isfinite(d.qpos).all():raise RuntimeError('non-finite')
mujoco.mj_forward(m,d);remaining=[]
for i,c in enumerate(d.contact[:d.ncon]):
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
 if 'ground' not in names and f[0]>.1:remaining.append({'geoms':names,'normal_force_n':float(f[0]),'distance_mm':float(c.dist*1000)})
row={'note':'Diagnostic only: suppressing these pairs does not establish that removing them is physically valid.',
     'removed_pairs':[list(x) for x in sorted(removed)],'remaining_high_self_contacts':remaining,
     'height_loss_mm':float((z0-d.xipos[body,2])*1000),
     'final_servo_error_deg':np.rad2deg(s.target_positions-d.actuator_length).tolist()}
(p/'targeted_collision_ablation.json').write_text(json.dumps(row,indent=2),encoding='utf8')
print('removed pairs',len(removed),'height loss mm',round(row['height_loss_mm'],3),
      'fifth error deg',round(row['final_servo_error_deg'][4],3),
      'max remaining self contact N',round(max([r['normal_force_n'] for r in remaining]+[0.]),2))
for r in sorted(remaining,key=lambda x:x['normal_force_n'],reverse=True)[:8]:print(r['geoms'],round(r['normal_force_n'],2),'N')
