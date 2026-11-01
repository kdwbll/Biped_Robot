from pathlib import Path
import xml.etree.ElementTree as E
import copy,mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_decomposed_tower_trial.xml');root=t.getroot();asset=root.find('asset');contact=root.find('contact');body=root.find(".//body[@name='robot_free']")
for i in range(12):
 old=f'hull_chassis_{i}';names=[]
 for f in (p/'chassis_decomposed').glob(f'chassis_{i}_*.stl'):
  name='dc_'+f.stem;names.append(name);E.SubElement(asset,'mesh',name=name,file='chassis_decomposed/'+f.name,scale='.001 .001 .001');E.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
 if not names:raise RuntimeError(old)
 for pair in list(contact.findall('pair')):
  if old in (pair.get('geom1'),pair.get('geom2')):
   contact.remove(pair)
   for name in names:
    cp=copy.deepcopy(pair)
    if cp.get('geom1')==old:cp.set('geom1',name)
    if cp.get('geom2')==old:cp.set('geom2',name)
    if cp.get('name'):cp.set('name',cp.get('name')+'_'+name)
    contact.append(cp)
root.set('model','biped_decomposed_chassis_trial');E.indent(root);out=p/'biped_decomposed_chassis_trial.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);bad=[]
for c in d.contact[:d.ncon]:
 if c.dist<-.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('pairs',m.npair,'initial contacts',d.ncon,'overlap >1mm',len(bad),'examples',bad[:8])
