from pathlib import Path
import xml.etree.ElementTree as E
import copy,mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_shaft_collision_trial.xml');root=t.getroot();assets=root.find('asset');contact=root.find('contact');hulls=[g.get('name') for g in root.iter('geom') if (g.get('name') or '').startswith('hull_chassis_')]
for leg in (0,1):
 old=f'upper_L{leg}_part0_0';body=root.find(f".//body[@name='L{leg}_part0']");names=[]
 for f in sorted((p/'tower_collision_pieces').glob(f'L{leg}_tower_*.stl')):
  name='tower_'+f.stem;names.append(name);E.SubElement(assets,'mesh',name=name,file='tower_collision_pieces/'+f.name,scale='.001 .001 .001');E.SubElement(body,'geom',name=name,type='mesh',mesh=name,mass='0',contype='0',conaffinity='0',rgba='0 0 0 0')
 for pair in list(contact.findall('pair')):
  if old in (pair.get('geom1'),pair.get('geom2')):
   contact.remove(pair)
   for name in names:
    cp=copy.deepcopy(pair)
    if cp.get('geom1')==old:cp.set('geom1',name)
    if cp.get('geom2')==old:cp.set('geom2',name)
    if cp.get('name'):cp.set('name',cp.get('name')+'_'+name)
    contact.append(cp)
 for a in names:
  for b in hulls:E.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','biped_decomposed_tower_trial');E.indent(root);out=p/'biped_decomposed_tower_trial.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);bad=[]
for c in d.contact[:d.ncon]:
 if c.dist<-.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('initial contacts',d.ncon,'overlap >1mm',len(bad),'examples',bad[:8])
