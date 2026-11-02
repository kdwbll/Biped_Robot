"""Prototype tangent clipping of proxy hulls near verified 20 mm CAD holes."""
from pathlib import Path
import json,collections
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
p=Path(__file__).parent
rows=json.loads((p/'chassis_witness_stl_containment_audit.json').read_text(encoding='utf8'))['results']
groups=collections.defaultdict(list)
for x in rows:
 if x['cad_nearest_surface_distance_mm']<=.1 or x['stl_inside']:continue
 center=np.array([42.,32. if x['leg']==0 else -73.]);point=np.array(x['point_assembly_mm'][:2]);delta=point-center;r=np.linalg.norm(delta)
 if 19.5<=r<=20.3:groups[(x['geoms'][0],x['leg'])].append(delta/r)
results=[]
for (name,leg),directions in sorted(groups.items()):
 center=np.array([42.,32. if leg==0 else -73.]);n=np.mean(directions,axis=0);n/=np.linalg.norm(n)
 path=p/'collision_finer/chassis0'/f"{name.removeprefix('nomerge_')}.stl";mesh=trimesh.load_mesh(path,process=True);v=mesh.vertices;hull=ConvexHull(v)
 signed=(v[:,:2]-center)@n-20.;kept=v[signed>=-1e-8].tolist();cut=[];edges=set()
 for f in hull.simplices:
  for i,j in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edges.add(tuple(sorted((int(i),int(j)))))
 for i,j in edges:
  if signed[i]*signed[j]<0:
   t=signed[i]/(signed[i]-signed[j]);point=(v[i]+t*(v[j]-v[i])).tolist();kept.append(point);cut.append(point)
 if len(kept)<4:raise RuntimeError((name,'clipping removed entire hull'))
 clipped=ConvexHull(np.array(kept));orig=hull.volume
 removed=np.vstack((v[signed<0],np.array(cut).reshape(-1,3)));max_r=float(np.max(np.linalg.norm(removed[:,:2]-center,axis=1))) if len(removed) else 20.
 results.append({'geom':name,'leg':leg,'witness_count':len(directions),'old_volume_mm3':float(orig),'new_volume_mm3':float(clipped.volume),'removed_percent':float(100*(orig-clipped.volume)/orig),'max_removed_vertex_radial_beyond_hole_mm':max(0.,max_r-20.),'tangent_unit':n.tolist()})
out={'prototype_only':True,'candidate_count':len(results),'results':results,'limitations':'One tangent halfspace inscribed outside an ideal 20 mm cylinder. It can remove genuine CAD material beyond the hole. Only witness-selected hulls examined, no collision model changed. Vertex-based removal metric bounds radial overshoot for the removed convex region but not other CAD features; dynamic validation absent.'}
(p/'tangent_clip_feasibility_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('candidate hulls',len(results),'max volume removed %',max(x['removed_percent'] for x in results),'max removed radial excess mm',max(x['max_removed_vertex_radial_beyond_hole_mm'] for x in results),flush=True)
