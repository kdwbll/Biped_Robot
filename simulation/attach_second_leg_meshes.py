import json,struct,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np,mujoco
p=Path(__file__).parent; axes=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).values())[1]; parts=sorted({v for x in axes for v in x['components']})
outdir=p/'meshes_second_leg_local';outdir.mkdir(exist_ok=True)
root=E.parse(p/'second_leg_constraint_experiment.xml'); m=root.getroot(); assets=E.Element('asset');m.insert(1,assets)
for i,part in enumerate(parts):
 matches=[f for f in (p/'meshes_original_coords').glob('*.STL') if f.stem.endswith(part)]
 if len(matches)!=1: raise RuntimeError(part)
 f=matches[0]; name=f'part{i}.stl'
 with f.open('rb') as h:
  header=h.read(80); n=struct.unpack('<I',h.read(4))[0]; a=np.frombuffer(h.read(),dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=n).copy()
 a['n'][:,1:]*=-1; a['v'][:,:,1]=-(a['v'][:,:,1]+41.0); a['v'][:,:,2]*=-1
 with (outdir/name).open('wb') as h: h.write(header);h.write(struct.pack('<I',n));h.write(a.tobytes())
 E.SubElement(assets,'mesh',name=f'mesh{i}',file='meshes_second_leg_local/'+name,scale='0.001 0.001 0.001')
 b=m.find(f"./worldbody/body[@name='part{i}']")
 for g in b.findall('geom'): g.set('rgba','0.9 0.3 0.1 0.3')
 E.SubElement(b,'geom',name='visual'+str(i),type='mesh',mesh=f'mesh{i}',contype='0',conaffinity='0',mass='0',rgba='0.65 0.72 0.85 1')
E.indent(m);out=p/'second_leg_mesh_experiment.xml';root.write(out,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(out)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
for _ in range(1000): mujoco.mj_step(model,data)
print('meshes',model.nmesh,'constraints',model.neq,'residual',max(abs(data.efc_pos)),'finite',np.isfinite(data.qpos).all())
