"""Verify exported sector hulls remain outside nominal cylindrical holes."""
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
p=Path(__file__).parent;data=json.loads((p/'sector_clip_prototype_audit.json').read_text())
rows=[]
for group in data['results']:
 center=np.array([42.,32. if group['leg']==0 else -73.])
 for filename in group['files']:
  mesh=trimesh.load_mesh(p/'collision_sector_trial/chassis0'/filename,process=True)
  if not mesh.is_watertight or mesh.volume<=0:raise RuntimeError((filename,'mesh invalid'))
  xy=np.unique(mesh.vertices[:,:2],axis=0);h=ConvexHull(xy);poly=xy[h.vertices];minimum=float('inf')
  for a,b in zip(poly,np.roll(poly,-1,axis=0)):
   v=b-a;denom=v@v
   t=np.clip(np.dot(center-a,v)/denom,0,1) if denom>0 else 0
   minimum=min(minimum,float(np.linalg.norm(a+t*v-center)))
  # A CCW convex hull with the origin inside contains the hole center.
  edge=np.roll(poly,-1,axis=0)-poly;relative=center-poly;cross=edge[:,0]*relative[:,1]-edge[:,1]*relative[:,0]
  if np.all(cross>=-1e-10):minimum=0.
  if minimum<19.99998:raise RuntimeError((filename,minimum))
  rows.append({'file':filename,'center_distance_min_mm':minimum,'volume_mm3':float(mesh.volume)})
out={'verified_piece_count':len(rows),'minimum_hole_radius_mm':min(x['center_distance_min_mm'] for x in rows),'results':rows,'limitations':'Checks exported convex polygon distance to assumed CAD cylinder axis in local XY, not full CAD-body containment or overlap between sectors.'}
(p/'sector_mesh_hole_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('verified pieces',len(rows),'minimum radius mm',out['minimum_hole_radius_mm'],flush=True)
