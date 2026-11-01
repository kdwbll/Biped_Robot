"""Controlled restoration of selected excluded contacts; no original model edit."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p = Path(__file__).parent
src = p / 'diagnostic_post_shaft_fix_gear_phase_25.xml'
groups = json.loads((p / 'pending_collision_source_audit.json').read_text(encoding='utf8'))['groups']
selected = [pair for g in groups if g['chassis_source_index'] == 5 for pair in g['proxy_pairs']]
tree = ET.parse(src)
contact = tree.getroot().find('contact')
existing = {tuple(sorted((x.get('geom1'), x.get('geom2')))) for x in contact.findall('pair')}
template = next(x for x in contact.findall('pair') if 'ground' not in (x.get('geom1'), x.get('geom2')))
for a, b in selected:
    if tuple(sorted((a,b))) in existing:
        raise RuntimeError('Already enabled')
    attrs = dict(template.attrib)
    attrs.pop('name', None)
    attrs.update(geom1=a, geom2=b)
    ET.SubElement(contact, 'pair', attrs)
tree.getroot().set('model', 'diagnostic_restore_chassis5_trial')
outmodel = p / 'diagnostic_restore_chassis5_trial.xml'
tree.write(outmodel, encoding='utf8', xml_declaration=True)
rows=[]
for filename in (src, outmodel):
    m=mujoco.MjModel.from_xml_path(str(filename));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
        kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
    ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
    start=d.xipos[body].copy();peak=0.;peak_pair=None;valid=True
    for step in range(round(3/m.opt.timestep)):
        if d.time >= 1.: servo.set_pulse_us([1500,1500,1500,1500,1510])
        old=d.time;servo.step()
        if not np.isfinite(d.qpos).all() or d.time < old + m.opt.timestep*.5:
            valid=False;break
        if step % 10 == 0:
            for i,c in enumerate(d.contact[:d.ncon]):
                if ground in (c.geom1,c.geom2): continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                if force[0] > peak:
                    peak=float(force[0]);peak_pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
    mujoco.mj_forward(m,d)
    row={'model':filename.name,'valid':valid,'time_s':float(d.time),'max_sampled_self_contact_n':peak,
         'peak_pair':peak_pair,'height_change_mm':float((d.xipos[body,2]-start[2])*1000),
         'hip_error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4]))}
    rows.append(row);print(row,flush=True)
(p/'collision_restore_chassis5_audit.json').write_text(json.dumps({'restored_pairs':selected,'results':rows,
 'limitations':'Initial source triangle separation does not prove solid separation; trial only, no full travel test. Contact force is sampled every 10 steps.'},indent=2),encoding='utf8')
