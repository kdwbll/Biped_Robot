"""Test nonoverlapping angular-sector clipping around a 20 mm hole on one hull."""
from pathlib import Path
import json,math
import numpy as np
import trimesh
from scipy.spatial import ConvexHull,QhullError
p=Path(__file__).parent
radius=20.;step=np.deg2rad(3.)
cad=json.loads((p/'cad_chassis0_holes_audit.json').read_text(encoding='utf8'))
if cad['document_unsaved'] or len(cad['faces'])!=2:raise RuntimeError('CAD hole evidence missing or unsaved')
for y in (32.,-73.):
 matches=[f for f in cad['faces'] if np.allclose(f['axis_point_mm'][:2],[42.,y],atol=.001) and abs(f['radius_mm']-radius)<.001 and np.allclose([f['bounds_mm'][2],f['bounds_mm'][5]],[136.,146.],atol=.001)]
 if len(matches)!=1:raise RuntimeError('CAD cylinder geometry differs from clipping assumptions')
def clip(vertices,n,offset):
 if len(vertices)<4:return None
 try:h=ConvexHull(vertices)
 except QhullError:return None
 signed=vertices[:,:2]@n-offset
 if np.all(signed<-1e-9):return None
 kept=vertices[signed>=-1e-9].tolist();edges=set()
 for f in h.simplices:
  for i,j in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edges.add(tuple(sorted((int(i),int(j)))))
 for i,j in edges:
  if signed[i]*signed[j]<0:
   t=signed[i]/(signed[i]-signed[j]);kept.append((vertices[i]+t*(vertices[j]-vertices[i])).tolist())
 if len(kept)<4:return None
 v=np.array(kept)
 try:h=ConvexHull(v)
 except QhullError:return None
 return v[h.vertices]
rows=json.loads((p/'chassis_witness_stl_containment_audit.json').read_text(encoding='utf8'))['results']
groups={}
for item in rows:
 if item['cad_nearest_surface_distance_mm']<=.1 or item['stl_inside']:continue
 center=np.array([42.,32. if item['leg']==0 else -73.]);delta=np.array(item['point_assembly_mm'][:2])-center;r=np.linalg.norm(delta)
 if 19.5<=r<=20.3:groups.setdefault((item['geoms'][0].removeprefix('nomerge_'),item['leg']),[]).append(delta/r)
results=[]
outdir=p/'collision_sector_trial/chassis0';outdir.mkdir(parents=True,exist_ok=True)
for (name,leg),directions in sorted(groups.items()):
 center=np.array([42.,32. if leg==0 else -73.]);mesh=trimesh.load_mesh(p/'collision_finer/chassis0'/f'{name}.stl',process=True);original=ConvexHull(mesh.vertices);v=mesh.vertices
 if v[:,2].min()<135.999 or v[:,2].max()>146.001:raise RuntimeError((name,'proxy extends beyond CAD cylinder'))
 direction=np.mean(directions,axis=0);anchor=np.arctan2(direction[1],direction[0]);angles=anchor+np.angle(np.exp(1j*(np.arctan2(v[:,1]-center[1],v[:,0]-center[0])-anchor)))
 lo=float(angles.min());hi=float(angles.max())
 if hi-lo>np.pi:raise RuntimeError((name,'Hull spans too large an angular range'))
 count=max(1,math.ceil((hi-lo)/step));edges=np.linspace(lo,hi,count+1);pieces=[];meshfiles=[]
 for a,b in zip(edges[:-1],edges[1:]):
  lower=np.array([-np.sin(a),np.cos(a)]);upper=np.array([np.sin(b),-np.cos(b)]);mid=(a+b)/2;normal=np.array([np.cos(mid),np.sin(mid)])
  cut=clip(v,lower,np.dot(lower,center))
  if cut is not None:cut=clip(cut,upper,np.dot(upper,center))
  if cut is not None:cut=clip(cut,normal,np.dot(normal,center)+radius)
  if cut is not None:
   h=ConvexHull(cut)
   if h.volume>1e-8:
    part=trimesh.Trimesh(vertices=cut,faces=h.simplices,process=True);part.fix_normals()
    if not part.is_watertight or part.volume<=0:raise RuntimeError((name,'invalid output piece'))
    filename=f'{name}_sector_{len(pieces):02}.stl';part.export(outdir/filename)
    pieces.append(h.volume);meshfiles.append(filename)
 if not pieces:raise RuntimeError((name,'No pieces'))
 bound=radius*(1/math.cos((hi-lo)/count/2)-1)
 results.append({'geom':name,'leg':leg,'angle_span_deg':float(np.rad2deg(hi-lo)),'sector_count':int(count),'nonempty_pieces':len(pieces),'files':meshfiles,'initial_volume_mm3':float(original.volume),'sum_piece_volumes_mm3':float(sum(pieces)),'relative_volume_removed_percent':float((original.volume-sum(pieces))/original.volume*100),'max_radial_removed_beyond_ideal_hole_mm':bound})
out={'candidate_hulls':len(results),'total_nonempty_pieces':sum(x['nonempty_pieces'] for x in results),'results':results,
 'limitations':'Mathematical bound for ideal 20 mm cylinder and 3 degree angular wedges, not CAD or mechanical verification. Witness-selected hulls only; new standalone meshes only; no XML or contact list changed, no dynamic tests. Adjacent pieces may cause redundant contacts.'}
(p/'sector_clip_prototype_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('hulls',len(results),'pieces',out['total_nonempty_pieces'],'maximum radial excess mm',max(x['max_radial_removed_beyond_ideal_hole_mm'] for x in results),'maximum removed volume %',max(x['relative_volume_removed_percent'] for x in results),flush=True)
