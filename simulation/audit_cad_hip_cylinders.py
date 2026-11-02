"""Inventory vertical cylindrical faces from temporary CAD body copies."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

p=Path(__file__).parent;pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open SimpleBiped2a_new.SLDASM in SolidWorks 2025')
 comps=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 out=[]
 for label,suffix in [('frame','16-40-03-958-1'),('leg0','16-17-29-589-1'),('leg1','16-17-29-589_1-1')]:
  comp=next(c for c in comps if c.Name2.endswith(suffix))
  for original in comp.GetModelDoc2.GetBodies2(0,True) or []:
   copy=original.Copy
   if callable(copy):copy=copy()
   if not copy.ApplyTransform(comp.Transform2):raise RuntimeError('Temporary body transform failed')
   volume=float(original.GetMassProperties(1)[3]*1e9)
   for idx,face in enumerate(copy.GetFaces() or []):
    surf=face.GetSurface
    if not surf.IsCylinder:continue
    a=list(surf.CylinderParams)
    if abs(a[5])<.99 or a[6]<.005:continue
    out.append({'component':label,'body_name':original.Name,'body_volume_mm3':volume,'face_index':idx,
      'axis_point_m':a[:3],'axis_direction':a[3:6],'radius_mm':a[6]*1000,'face_area_mm2':float(face.GetArea*1e6)})
 (p/'cad_hip_cylinders_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 for row in out:
  print(row['component'],round(row['body_volume_mm3']), 'center mm',[round(x*1000,2) for x in row['axis_point_m'][:2]],'radius mm',round(row['radius_mm'],3),'area',round(row['face_area_mm2']),flush=True)
finally:pythoncom.CoUninitialize()
