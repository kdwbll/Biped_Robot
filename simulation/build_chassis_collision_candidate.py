from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_mass_knee_candidate.xml');root=t.getroot();asset=root.find('asset');contact=root.find('contact');body=root.find(".//body[@name='robot_free']")
for f in (p/'chassis_collision_pieces').glob('*.stl'):
 name='hull_'+f.stem;E.SubElement(asset,'mesh',name=name,file='chassis_collision_pieces/'+f.name,scale='0.001 0.001 0.001');E.SubElement(body,'geom',name=name,type='mesh',mesh=name,mass='0',contype='0',conaffinity='0',rgba='0 0 0 0')
 for g in root.iter('geom'):
  n=g.get('name','')
  if n.startswith(('col_L','knee_L')):E.SubElement(contact,'pair',geom1=name,geom2=n,solref='0.001 1',solimp='0.99 0.999 0.001')
root.set('model','biped_chassis_collision_candidate');E.indent(root);out=p/'biped_chassis_collision_candidate.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d);bad=[]
for i in range(d.ncon):
 c=d.contact[i]
 if c.dist<-.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('initial contacts',d.ncon,'overlap >1mm',len(bad),'examples',bad[:8])
