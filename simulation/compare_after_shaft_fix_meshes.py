"""Compare current component STLs with previous exports in assembly coordinates."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import trimesh

p=Path(__file__).parent
old=p/'meshes_original_coords';new=p/'meshes_after_shaft_fix';rows=[]
for f in sorted(new.glob('*.stl')):
 prior=old/f.name
 if not prior.exists():continue
 a=trimesh.load_mesh(prior,process=True);b=trimesh.load_mesh(f,process=True)
 da=cKDTree(b.vertices).query(a.vertices,workers=1)[0]
 db=cKDTree(a.vertices).query(b.vertices,workers=1)[0]
 row={'name':f.name,'old_vertices':len(a.vertices),'new_vertices':len(b.vertices),
      'old_faces':len(a.faces),'new_faces':len(b.faces),'old_bounds_mm':a.bounds.tolist(),'new_bounds_mm':b.bounds.tolist(),
      'symmetric_max_nearest_vertex_mm':float(max(np.max(da),np.max(db))),
      'symmetric_rms_nearest_vertex_mm':float(np.sqrt((np.sum(da*da)+np.sum(db*db))/(len(da)+len(db))))}
 rows.append(row)
 print(f.name.split(' ')[-1], 'max mm',round(row['symmetric_max_nearest_vertex_mm'],4),
       'rms mm',round(row['symmetric_rms_nearest_vertex_mm'],4),flush=True)
(p/'post_shaft_fix_mesh_comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
