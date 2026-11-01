from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_estimated_mass_collision.xml');root=t.getroot();asset=root.find('asset');contact=root.find('contact');g={}
for leg in (0,1):
 body=root.find(f".//body[@name='L{leg}_part3']");g[leg]=[]
 for f in sorted((p/'knee_collision_pieces').glob(f'L{leg}_part3_*.stl')):
  name='knee_'+f.stem;g[leg].append(name);E.SubElement(asset,'mesh',name=name,file='knee_collision_pieces/'+f.name,scale='0.001 0.001 0.001');E.SubElement(body,'geom',name=name,type='mesh',mesh=name,mass='0',contype='0',conaffinity='0',rgba='0 0 0 0')
for leg in (0,1):
 for x in g[leg]:
  for y in root.iter('geom'):
   name=y.get('name','')
   if name.startswith(f'col_L{leg}_part6_') or name.startswith(f'col_L{1-leg}_part'):
    E.SubElement(contact,'pair',geom1=x,geom2=name,solref='0.001 1',solimp='0.99 0.999 0.001')
for x in g[0]:
 for y in g[1]:E.SubElement(contact,'pair',geom1=x,geom2=y,solref='0.001 1',solimp='0.99 0.999 0.001')
root.set('model','biped_mass_knee_candidate');E.indent(root);out=p/'biped_mass_knee_candidate.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d);print('knee pieces',list(map(len,g.values())),'initial contacts',d.ncon)
bad=[]
for i in range(d.ncon):
 c=d.contact[i]
 if c.dist<-.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('initial overlap >1mm',len(bad),bad[:8])
