from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_centered_initial_pose.xml');root=t.getroot();asset=root.find('asset');contact=root.find('contact');groups={0:[],1:[]}
for leg in (0,1):
 for f in (p/'upper_collision_pieces').glob(f'L{leg}_part0_*.stl'):
  name='upper_'+f.stem;groups[leg].append(name);E.SubElement(asset,'mesh',name=name,file='upper_collision_pieces/'+f.name,scale='.001 .001 .001');b=root.find(f".//body[@name='L{leg}_part0']");E.SubElement(b,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
 for name in groups[leg]:
  for g in root.iter('geom'):
   n=g.get('name',''); own_ok=any(n.startswith(f'col_L{leg}_part{j}_') for j in (4,5,6)) or n.startswith(f'knee_L{leg}_part3_');other_ok=n.startswith((f'col_L{1-leg}_',f'knee_L{1-leg}_'))
   if own_ok or other_ok:E.SubElement(contact,'pair',geom1=name,geom2=n,solref='.001 1',solimp='.99 .999 .001')
root.set('model','biped_upper_collision_trial');E.indent(root);out=p/'biped_upper_collision_trial.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);bad=[]
for i in range(d.ncon):
 c=d.contact[i];a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
 if c.dist<-.001:bad.append((a,b,round(-c.dist*1000,2)))
print('upper pieces',list(map(len,groups.values())),'initial contacts',d.ncon,'overlap > 1mm',len(bad),bad[:10])
