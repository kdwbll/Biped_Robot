"""Compare peak hip-turn proxy contact to source triangles at the same pose."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
 kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,10])
for step in range(round(6/m.opt.timestep)):
 if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1510])
 old=d.time;servo.step()
 if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:raise RuntimeError('Numerical failure')
mujoco.mj_forward(m,d)
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
chassis=[c for c in source.split(only_watertight=False,repair=False) if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001][0]
chassis.apply_scale(.001)
root=ET.parse(p/'second_leg_mesh_experiment.xml').getroot()
parts=trimesh.load_mesh(p/root.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
tower=max((x for x in parts if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001)
a=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'dc_chassis_0_7')
b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'tower_L1_tower_8')
def pose(body):
 t=np.eye(4);t[:3,:3]=d.xmat[body].reshape(3,3);t[:3,3]=d.xpos[body];return t
cm=trimesh.collision.CollisionManager();cm.add_object('chassis',chassis,transform=pose(m.geom_bodyid[a]))
collides,details=cm.in_collision_single(tower,transform=pose(m.geom_bodyid[b]),return_data=True)
out={'time_s':float(d.time),'proxy_pair':['dc_chassis_0_7','tower_L1_tower_8'],
 'proxy_signed_distance_mm':float(mujoco.mj_geomDistance(m,d,a,b,1,None)*1000),
 'source_triangles_intersect':bool(collides),'source_contact_count':len(details),
 'source_surface_distance_mm':float(cm.min_distance_single(tower,transform=pose(m.geom_bodyid[b]))*1000),
 'limitations':'Source STL triangle collision is not an exact solid CAD interference or validation of individual proxy contact.'}
(p/'hip_peak_source_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
