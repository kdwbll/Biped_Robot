"""Build finer experimental decompositions without replacing previous meshes."""
from pathlib import Path
import json
import sys
import time
import xml.etree.ElementTree as ET
import trimesh
import coacd

p=Path(__file__).parent
target=sys.argv[1]
if target=='chassis5':
    source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
    parts=[c for c in source.split(only_watertight=False,repair=False) if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001]
    mesh=parts[5]
elif target in ('L0','L1'):
    root=ET.parse(p/('single_leg_mesh_experiment.xml' if target=='L0' else 'second_leg_mesh_experiment.xml')).getroot()
    parts=trimesh.load_mesh(p/root.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
    mesh=max((x for x in parts if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces))
else: raise ValueError('Expected chassis5, L0 or L1')
out=p/'collision_highres'/target;out.mkdir(parents=True,exist_ok=True)
coacd.set_log_level('error');start=time.monotonic()
result=coacd.run_coacd(coacd.Mesh(mesh.vertices.astype('float64'),mesh.faces.astype('int32')),
      threshold=.008,max_convex_hull=96,resolution=800,mcts_iterations=35,preprocess_mode='off')
for i,(vertices,faces) in enumerate(result):
    trimesh.Trimesh(vertices,faces,process=False).export(out/f'{target}_{i:03}.stl')
summary={'target':target,'source_faces':len(mesh.faces),'threshold':.008,'max_convex_hull':96,
         'output_count':len(result),'seconds':time.monotonic()-start,'units':'mm'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(summary,flush=True)
