"""Estimate gear tooth periodicity from outer STL vertices; geometry approximation only."""
from pathlib import Path
import json
import math
import numpy as np
from scipy.spatial import cKDTree
import trimesh

p=Path(__file__).parent
specs=[('pinion',p/'gear_phase_visuals'/'pinion_local.stl',(0.,0.)),
       ('large0',next((p/'meshes_after_shaft_fix').glob('*16-17-29-589-1.STL')),(42.,32.)),
       ('large1',next((p/'meshes_after_shaft_fix').glob('*16-17-29-589_1-1.STL')),(42.,-73.))]
out={}
for name,path,center in specs:
 mesh=trimesh.load_mesh(path,process=True)
 if name!='pinion':
  parts=mesh.split(only_watertight=False,repair=False)
  choices=[x for x in parts if abs(abs(x.volume)-81650)<1000]
  if len(choices)!=1:raise RuntimeError((name,len(choices)))
  mesh=choices[0]
 xy=mesh.vertices[:,:2]-np.array(center)
 radius=np.linalg.norm(xy,axis=1)
 points=np.column_stack((xy,mesh.vertices[:,2]))[radius>np.max(radius)*.93]
 tree=cKDTree(points);rows=[]
 for n in range(15,65):
  angle=2*math.pi/n;c=math.cos(angle);s=math.sin(angle)
  rot=points.copy();rot[:,:2]=points[:,:2]@np.array([[c,s],[-s,c]])
  dist=tree.query(rot)[0]
  rows.append({'candidate_teeth':n,'nearest_vertex_rms_mm':float(np.sqrt(np.mean(dist**2))),
               'nearest_vertex_p95_mm':float(np.percentile(dist,95))})
 rows.sort(key=lambda x:x['nearest_vertex_rms_mm'])
 out[name]={'source':str(path),'outer_vertices':len(points),'max_radius_mm':float(max(radius)),
            'best_candidates':rows[:10],'all_candidates':rows}
 print(name,'best',[(r['candidate_teeth'],round(r['nearest_vertex_rms_mm'],4)) for r in rows[:6]],flush=True)
(p/'gear_tooth_periodicity_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
