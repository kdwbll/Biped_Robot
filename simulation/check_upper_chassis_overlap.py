import xml.etree.ElementTree as E,mujoco
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_shaft_collision_trial.xml');r=t.getroot();c=r.find('contact');names=[g.get('name','') for g in r.iter('geom')];ups=[n for n in names if n.startswith('upper_L')];hulls=[n for n in names if n.startswith('hull_chassis_')]
for a in ups:
 for b in hulls:E.SubElement(c,'pair',geom1=a,geom2=b,solref='.001 1')
out=p/'temp_upper_chassis_collision.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);print('contacts',d.ncon)
for c in d.contact[:d.ncon]:
 if c.dist < -.001:print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2),'mm')
