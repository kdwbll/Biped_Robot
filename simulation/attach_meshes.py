import json,xml.etree.ElementTree as E
from pathlib import Path
import mujoco,numpy as np
p=Path(__file__).parent; axes=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).values())[0]
parts=sorted({v for x in axes for v in x['components']}); root=E.parse(p/'single_leg_constraint_experiment.xml'); m=root.getroot(); assets=E.Element('asset'); m.insert(1,assets)
for i,part in enumerate(parts):
 matches=[f for f in (p/'meshes_original_coords').glob('*.STL') if f.stem.endswith(part)]
 if len(matches)!=1: raise RuntimeError((part,matches))
 f=matches[0]; name=f'mesh{i}'
 E.SubElement(assets,'mesh',name=name,file=str(f.relative_to(p)).replace('\\','/'),scale='0.001 0.001 0.001')
 b=m.find(f"./worldbody/body[@name='part{i}']")
 for g in b.findall('geom'): g.set('rgba','0.9 0.3 0.1 0.3')
 E.SubElement(b,'geom',name='visual'+str(i),type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0.65 0.72 0.85 1')
E.indent(m); out=p/'single_leg_mesh_experiment.xml'; root.write(out,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(out)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
for _ in range(1000): mujoco.mj_step(model,data)
print('meshes',model.nmesh,'bodies',model.nbody-1,'constraint residual',max(abs(data.efc_pos)),'finite',np.isfinite(data.qpos).all())
