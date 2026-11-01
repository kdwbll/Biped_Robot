from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
p=Path(r'C:\Practice_project\biped_sim');src=p/'biped_decomposed_chassis_trial.xml';m=mujoco.MjModel.from_xml_path(str(src));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
exclude=set()
for c in d.contact[:d.ncon]:
 a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
 if c.dist<-.0005 and ((a.startswith('dc_chassis_') and b.startswith('tower_')) or (b.startswith('dc_chassis_') and a.startswith('tower_'))): exclude.add(frozenset((a,b)))
t=E.parse(src);root=t.getroot();contact=root.find('contact');removed=0
for pair in list(contact.findall('pair')):
 if frozenset((pair.get('geom1'),pair.get('geom2'))) in exclude:contact.remove(pair);removed+=1
root.set('model','biped_filtered_collision_experimental');E.indent(root);out=p/'biped_filtered_collision_experimental.xml';t.write(out,encoding='utf8',xml_declaration=True)
m2=mujoco.MjModel.from_xml_path(str(out));d2=mujoco.MjData(m2);mujoco.mj_resetDataKeyframe(m2,d2,0);mujoco.mj_forward(m2,d2)
print('excluded colliding tower-chassis pairs',removed,'of',m.npair,'remaining',m2.npair,'initial contacts',d2.ncon,'minimum distance mm',round(min([c.dist for c in d2.contact[:d2.ncon]]+[0])*1000,3))
