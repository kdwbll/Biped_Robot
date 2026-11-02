"""Trial fully refined chassis-5/tower contact pairs; preserves original XML."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
tree=ET.parse(p/'diagnostic_post_shaft_fix_gear_phase_25.xml');root=tree.getroot()
assets=root.find('asset');contact=root.find('contact')
parent=next(b for b in root.iter('body') if b.find("geom[@name='dc_chassis_5_0']") is not None)
new=[]
for file in sorted((p/'chassis5_collision_refined').glob('*.stl')):
    name='refined_'+file.stem;new.append(name)
    ET.SubElement(assets,'mesh',name=name,file='chassis5_collision_refined/'+file.name,scale='.001 .001 .001')
    ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
removed=0
for pair in list(contact.findall('pair')):
    a,b=pair.get('geom1'),pair.get('geom2')
    if (a.startswith('dc_chassis_5_') and (b.startswith('tower_') or b.startswith('refined_L'))) or (b.startswith('dc_chassis_5_') and (a.startswith('tower_') or a.startswith('refined_L'))):
        contact.remove(pair);removed+=1
towers=[g.get('name') for g in root.iter('geom') if (g.get('name') or '').startswith('refined_L')]
for a in new:
    for b in towers:
        ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','diagnostic_refined_chassis5_trial')
path=p/'diagnostic_refined_chassis5_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
rows=[]
for filename in (p/'diagnostic_post_shaft_fix_gear_phase_25.xml',path):
    m=mujoco.MjModel.from_xml_path(str(filename));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
         kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
    ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
    z0=d.xipos[body,2];peak=0.;peakpair=None;valid=True
    for step in range(round(1/m.opt.timestep)):
        old=d.time;servo.step()
        if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:valid=False;break
        if step%10==0:
            for i,c in enumerate(d.contact[:d.ncon]):
                if ground in (c.geom1,c.geom2):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                if f[0]>peak:
                    peak=float(f[0]);peakpair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
    mujoco.mj_forward(m,d)
    row={'model':filename.name,'valid':valid,'duration_s':float(d.time),'height_change_mm':float((d.xipos[body,2]-z0)*1000),
         'sampled_peak_self_force_n':peak,'peak_pair':peakpair}
    rows.append(row);print(row,flush=True)
(p/'refined_chassis5_trial_audit.json').write_text(json.dumps({'old_pairs_removed':removed,'new_pairs':len(new)*len(towers),
    'results':rows,'limitations':'Experimental convex decomposition; contacts sampled every 10 steps, no moving range or solid CAD proof.'},indent=2),encoding='utf8')
