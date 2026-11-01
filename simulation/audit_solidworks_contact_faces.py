"""Inspect CAD faces at exact closest-distance points in open SolidWorks assembly."""
from pathlib import Path
import json
import numpy as np
import pythoncom
import win32com.client as w

p=Path(__file__).parent;closest=json.loads((p/'solidworks_closest_distance_audit.json').read_text(encoding='utf8'))
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved assembly')
 comps=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 def body(suffix,volume):
  comp=next(c for c in comps if c.Name2.endswith(suffix))
  matches=[b for b in (comp.GetModelDoc2.GetBodies2(0,True) or []) if abs(b.GetMassProperties(1)[3]*1e9-volume)<200]
  if len(matches)!=1:raise RuntimeError((suffix,volume,len(matches)))
  result=matches[0].Copy
  if callable(result):result=result()
  if not result.ApplyTransform(comp.Transform2):raise RuntimeError('Failed to transform temporary body copy')
  return result
 torso={0:body('16-40-03-958-1',135648.8),5:body('16-40-03-958-1',143768.7)}
 legs={0:body('16-17-29-589-1',129952.3),1:body('16-17-29-589_1-1',129952.3)}
 def faces_near(solid,point):
  point=np.asarray(point);out=[]
  for i,face in enumerate(solid.GetFaces() or []):
   result=face.GetClosestPointOn(*map(float,point))
   if result is None:continue
   gap=float(np.linalg.norm(np.asarray(result[:3])-point))*1000
   if gap>.01:continue
   normal=list(face.Normal or (0.,0.,0.));surface=face.GetSurface
   out.append({'face_index':i,'closest_point_gap_mm':gap,'planar':bool(surface.IsPlane),
               'cylindrical':bool(surface.IsCylinder),'planar_normal':normal,'face_area_mm2':float(face.GetArea*1e6)})
  return out
 rows=[]
 for r in closest['results']:
  row={'chassis_source_index':r['chassis_source_index'],'leg':r['leg'],
       'chassis_faces':faces_near(torso[r['chassis_source_index']],r['point1_m']),
       'tower_faces':faces_near(legs[r['leg']],r['point2_m'])}
  rows.append(row);print(row,flush=True)
 out={'results':rows,'limit_mm':.01,'limitations':'Faces containing each one closest point only; normals are planar-face normals in API frame and do not identify full contact patch or constrain movement.'}
 (p/'solidworks_contact_faces_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
finally:pythoncom.CoUninitialize()
