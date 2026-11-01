from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_ground_contact_candidate.xml');c=t.getroot().find('contact')
for leg in (0,1):
 for i in range(1,7):
  for j in range(i+1,7): E.SubElement(c,'pair',geom1=f'L{leg}_visual{i}',geom2=f'L{leg}_visual{j}')
for i in range(1,7):
 for j in range(1,7):
  if (i,j)!=(6,6):E.SubElement(c,'pair',geom1=f'L0_visual{i}',geom2=f'L1_visual{j}')
out=p/'temp_collision_feasibility.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
print('initial contacts',d.ncon)
for i in range(d.ncon):
 x=d.contact[i]
 if x.dist < -.001:print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,x.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,x.geom2),'penetration mm',round(-x.dist*1000,2))
