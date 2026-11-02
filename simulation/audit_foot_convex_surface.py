"""Compare raw sole lower surfaces with MuJoCo mesh convex-hull approximation."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_foot_suppressed_plate_trial.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)

def lower_surface(v,faces,points):
 tri=v[faces];a=tri[:,0,:2];e1=tri[:,1,:2]-a;e2=tri[:,2,:2]-a
 det=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0];valid=np.abs(det)>1e-13
 a=a[valid];e1=e1[valid];e2=e2[valid];det=det[valid];zz=tri[valid,:,2]
 result=np.full(len(points),np.nan)
 for start in range(0,len(points),32):
  pp=points[start:start+32,None,:]-a[None,:,:]
  u=(pp[:,:,0]*e2[None,:,1]-pp[:,:,1]*e2[None,:,0])/det[None,:]
  w=(e1[None,:,0]*pp[:,:,1]-e1[None,:,1]*pp[:,:,0])/det[None,:]
  inside=(u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)
  z=zz[None,:,0]+u*(zz[None,:,1]-zz[None,:,0])+w*(zz[None,:,2]-zz[None,:,0])
  values=np.min(np.where(inside,z,np.inf),axis=1);values[~np.isfinite(values)]=np.nan;result[start:start+len(values)]=values
 return result
rows=[]
for leg in (0,1):
 g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{leg}_visual6');mid=m.geom_dataid[g];va=m.mesh_vertadr[mid];fa=m.mesh_faceadr[mid]
 base=m.mesh_vert[va:va+m.mesh_vertnum[mid]]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
 faces=m.mesh_face[fa:fa+m.mesh_facenum[mid]]
 for axis,angle in [('initial',0),('x',-15),('x',-5),('x',5),('x',15),('y',-15),('y',-5),('y',5),('y',15)]:
  t=np.deg2rad(angle);c=np.cos(t);ss=np.sin(t)
  rotation=np.array([[1,0,0],[0,c,-ss],[0,ss,c]]) if axis=='x' else np.array([[c,0,ss],[0,1,0],[-ss,0,c]]) if axis=='y' else np.eye(3)
  pivot=base.mean(axis=0);v=(base-pivot)@rotation.T+pivot;v[:,2]-=v[:,2].min()
  hull=ConvexHull(v);xmin,ymin=v[:,:2].min(axis=0);xmax,ymax=v[:,:2].max(axis=0)
  x=np.linspace(xmin+1e-6,xmax-1e-6,81);y=np.linspace(ymin+1e-6,ymax-1e-6,49);xx,yy=np.meshgrid(x,y);pts=np.column_stack((xx.ravel(),yy.ravel()))
  actual=lower_surface(v,faces,pts);convex=lower_surface(v,hull.simplices,pts);both=np.isfinite(actual)&np.isfinite(convex);gap=(actual[both]-convex[both])*1000
  near={}
  for height in (0.5,1.,2.,5.):
   mask=both & (convex*1000<=height);diff=(actual[mask]-convex[mask])*1000
   hull_only=(np.isfinite(convex)&~np.isfinite(actual)&(convex*1000<=height))
   near[str(height)]={'sample_count':int(mask.sum()),'max_gap_mm':float(diff.max()) if len(diff) else None,'count_gap_over_0_5mm':int((diff>.5).sum()),'hull_only_count':int(hull_only.sum())}
  row={'leg':leg,'rotation_axis':axis,'rotation_deg':angle,'near_ground_height_mm':near,'samples_grid':[81,49], 'hull_only_total':int((np.isfinite(convex)&~np.isfinite(actual)).sum()),'max_gap_mm':float(np.max(gap))}
  rows.append(row);print('leg',leg,'axis',axis,'deg',angle,'near1mm',near['1.0'],flush=True)
out={'model':'diagnostic_foot_suppressed_plate_trial.xml','pose':'initial and isolated geometric foot tilts; lowest vertex aligned with ground','results':rows,'limitations':'Uniform world-XY grid samples compare original triangles and their mathematical convex hull, not actual MuJoCo contact forces. Tilts are geometric, not necessarily reachable joint poses. Ground flat; lateral and self-collision are excluded. CAD meshes and physical contacts remain estimates.'}
(p/'foot_convex_surface_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
