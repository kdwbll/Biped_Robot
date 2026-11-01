from pathlib import Path
import xml.etree.ElementTree as E
import json,mujoco
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_ground_contact_candidate.xml');root=t.getroot();root.set('model','biped_partial_collision_v2');asset=root.find('asset');contact=root.find('contact');g={}
for f in (p/'collision_pieces').glob('*.stl'):
 name='col_'+f.stem;leg=int(f.stem[1]);part=int(f.stem.split('_')[1][4:]);g.setdefault((leg,part),[]).append(name)
 E.SubElement(asset,'mesh',name=name,file='collision_pieces/'+f.name,scale='0.001 0.001 0.001')
 b=root.find(f".//body[@name='L{leg}_part{part}']");E.SubElement(b,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
axes=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).values());linked=[]
for leg in (0,1):
 parts=sorted({x for a in axes[leg] for x in a['components']});linked.append({frozenset(parts.index(x) for x in a['components']) for a in axes[leg]})
keys=list(g)
for i,(leg,a) in enumerate(keys):
 for leg2,b in keys[i+1:]:
  if leg==leg2 and frozenset((a,b)) in linked[leg]:continue
  for x in g[(leg,a)]:
   for y in g[(leg2,b)]:E.SubElement(contact,'pair',geom1=x,geom2=y,solref='0.001 1')
out=p/'self_collision_split_candidate.xml';E.indent(root);t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
print('pieces',sum(map(len,g.values())),'contacts',d.ncon)
bad=[]
for i in range(d.ncon):
 c=d.contact[i]
 if c.dist < -.001:bad.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(-c.dist*1000,2)))
print('initial penetrations > 1mm',len(bad));print(bad[:12])

