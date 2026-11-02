"""Map selected positive-clearance proxy witnesses to open SolidWorks 2025 faces."""
from pathlib import Path
import json
import mujoco
import numpy as np
import pythoncom
import win32com.client as w
p=Path(__file__).parent
model=mujoco.MjModel.from_xml_path(str(p/'diagnostic_restored_near_pairs_trial.xml'));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
raw=json.loads((p/'chassis0_finer_proxy_range_audit.json').read_text(encoding='utf8'))['near_pairs']
choices=[]
for leg in (0,1):
 candidates=sorted([x for x in raw if x['geoms'][1].startswith(f'nomerge_L{leg}_') and x['min_mm']<=.05 and .0001<x['initial_mm']<.1],key=lambda x:x['initial_mm'])
 if not candidates:raise RuntimeError('No selected proxies')
 choices.extend((leg,x) for x in candidates)
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33');doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open SolidWorks 2025 assembly')
 comps=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 def getcopy(suffix,volume):
  comp=next(c for c in comps if c.Name2.endswith(suffix));bs=[b for b in (comp.GetModelDoc2.GetBodies2(0,True) or []) if abs(b.GetMassProperties(1)[3]*1e9-volume)<200]
  if len(bs)!=1:raise RuntimeError((suffix,len(bs)))
  copy=bs[0].Copy
  if callable(copy):copy=copy()
  if not copy.ApplyTransform(comp.Transform2):raise RuntimeError('Body transform failed')
  return copy
 chassis=getcopy('16-40-03-958-1',135648.8)
 towers=[getcopy('16-17-29-589-1',129952.3),getcopy('16-17-29-589_1-1',129952.3)]
 faces=[list(chassis.GetFaces()),*[list(t.GetFaces()) for t in towers]]
 def nearest(fs,pt):
  best=None
  for i,face in enumerate(fs):
   q=face.GetClosestPointOn(*map(float,pt))
   if q is None:continue
   xyz=np.asarray(q[:3]);distance=float(np.linalg.norm(xyz-pt)*1000)
   if best is None or distance<best['gap_mm']:
    surface=face.GetSurface
    best={'face_index':i,'gap_mm':distance,'planar':bool(surface.IsPlane),'cylindrical':bool(surface.IsCylinder),'area_mm2':float(face.GetArea*1e6),'normal':list(face.Normal or (0.,0.,0.))}
  if best is None:raise RuntimeError('No faces evaluated')
  return best
 root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot_free');R=data.xmat[root].reshape(3,3);pos=data.xpos[root];rows=[]
 for leg,row in choices:
  ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in row['geoms']];ends=np.zeros(6);gap=mujoco.mj_geomDistance(model,data,*ids,.003,ends)
  if gap<=0:raise RuntimeError('Expected separated witness points')
  cad=[(ends[:3]-pos)@R,(ends[3:]-pos)@R]
  a=nearest(faces[0],cad[0]);b=nearest(faces[leg+1],cad[1])
  out={'leg':leg,'geoms':row['geoms'],'proxy_separation_mm':gap*1000,'proxy_min_sampled_mm':row['min_mm'],'chassis_cad_face':a,'tower_cad_face':b}
  rows.append(out);
  if len(rows)%25==0:print('evaluated',len(rows),'pairs',flush=True)
 result={'selection':'All remaining proxy pairs with initial clearance in (0.0001, 0.1) mm and sampled minimum <=0.05 mm','results':rows,'limitations':'Only the initial keyframe and selected non-overlapping proxy pairs. Witness to nearest CAD face is not the CAD body-to-body clearance or proof of collision. CAD-frame correspondence relies on previous approximate mapping; angles and CAD configuration may differ. Pairs not selected are not assessed.'}
 (p/'remaining_collision_cad_witness_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
finally:pythoncom.CoUninitialize()
