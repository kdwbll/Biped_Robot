"""Bounded CoACD experiment without merging convex pieces.

CoACD's hull count cap is a merge constraint, not a local accuracy guarantee.
This generates independent diagnostic geometry and never edits the baseline.
"""
from pathlib import Path
import json
import sys
import time
import trimesh
import coacd

p=Path(__file__).parent
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[c for c in source.split(only_watertight=False,repair=False) if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001]
mesh=parts[5];out=p/'collision_nomerge'/'chassis5';out.mkdir(parents=True,exist_ok=True)
coacd.set_log_level('error');start=time.monotonic()
result=coacd.run_coacd(coacd.Mesh(mesh.vertices.astype('float64'),mesh.faces.astype('int32')),
 threshold=.008,max_convex_hull=-1,merge=False,resolution=500,mcts_iterations=15,preprocess_mode='off')
for i,(vertices,faces) in enumerate(result):
 trimesh.Trimesh(vertices,faces,process=False).export(out/f'chassis5_{i:03}.stl')
summary={'target':'chassis5','source_faces':len(mesh.faces),'threshold':.008,'merge':False,
 'resolution':500,'mcts_iterations':15,'output_count':len(result),'seconds':time.monotonic()-start,'units':'mm'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(summary,flush=True)
