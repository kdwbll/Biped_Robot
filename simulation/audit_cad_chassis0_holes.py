"""Read nominal chassis-0 bearing hole axes and axial bounds in SolidWorks 2025."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w
p=Path(__file__).parent;pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33');doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved SolidWorks assembly')
 c=next(c for c in doc.GetComponents(False) if c.Name2.endswith('16-40-03-958-1'))
 matches=[b for b in c.GetModelDoc2.GetBodies2(0,True) if abs(b.GetMassProperties(1)[3]*1e9-135648.8)<200]
 if len(matches)!=1:raise RuntimeError('Chassis body not unique')
 copy=matches[0].Copy
 if callable(copy):copy=copy()
 if not copy.ApplyTransform(c.Transform2):raise RuntimeError('Transform failed')
 faces=[]
 for face in copy.GetFaces():
  surf=face.GetSurface
  if not surf.IsCylinder:continue
  params=surf.CylinderParams
  if abs(params[6]*1000-20)>.001:continue
  box=face.GetBox
  if callable(box):box=box()
  faces.append({'axis_point_mm':[x*1000 for x in params[:3]],'axis_direction':list(params[3:6]),'radius_mm':params[6]*1000,'bounds_mm':[x*1000 for x in box],'area_mm2':face.GetArea*1e6})
 if len(faces)!=2:raise RuntimeError('Expected two hole cylinders')
 out={'document':doc.GetPathName,'document_unsaved':bool(doc.GetSaveFlag),'faces':faces,'limitations':'Current CAD nominal geometry only. Does not establish manufacturing tolerances, material, actual support loading, or collision validation.'}
 (p/'cad_chassis0_holes_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('hole faces',len(faces),'document unsaved',out['document_unsaved'],flush=True)
finally:pythoncom.CoUninitialize()
