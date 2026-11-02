"""Decompose source chassis solid 5 for isolated collision-proxy experiments."""
from pathlib import Path
import json
import time
import trimesh
import coacd

p=Path(__file__).parent
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
pieces=[c for c in source.split(only_watertight=False,repair=False)
        if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001]
mesh=pieces[5]
out=p/'chassis5_collision_refined';out.mkdir(exist_ok=True)
coacd.set_log_level('error');start=time.monotonic()
result=coacd.run_coacd(coacd.Mesh(mesh.vertices.astype('float64'),mesh.faces.astype('int32')),
    threshold=.025,max_convex_hull=32,resolution=500,mcts_iterations=30,preprocess_mode='off')
for i,(vertices,faces) in enumerate(result):
    trimesh.Trimesh(vertices,faces,process=False).export(out/f'chassis5_{i:02}.stl')
summary={'source_piece':5,'faces':len(mesh.faces),'output_count':len(result),'seconds':time.monotonic()-start,
         'note':'Source-coordinate meshes in mm; diagnostic only, no collision coverage claim.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(summary,flush=True)
