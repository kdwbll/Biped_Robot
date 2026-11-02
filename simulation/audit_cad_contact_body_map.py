"""Map collision source meshes to SolidWorks 2025 bodies by volume and bounds.

Uses existing application, copies bodies and never changes or closes the document.
"""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

p=Path(__file__).parent
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved SimpleBiped2a_new.SLDASM in SolidWorks 2025')
 components=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 out={}
 for label,suffix in [('torso','16-40-03-958-1'),('leg0','16-17-29-589-1'),('leg1','16-17-29-589_1-1')]:
  comp=next(c for c in components if c.Name2.endswith(suffix))
  bodies=[]
  for body in comp.GetModelDoc2.GetBodies2(0,True) or []:
   cp=body.Copy
   if callable(cp):cp=cp()
   if not cp.ApplyTransform(comp.Transform2):raise RuntimeError('Body copy transform failed')
   box=list(cp.GetBodyBox())
   bodies.append({'name':body.Name,'volume_mm3':float(body.GetMassProperties(1)[3]*1e9),
                  'assembly_bounds_mm':[float(x*1000) for x in box]})
  out[label]=bodies
 (p/'cad_contact_body_map.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 for k,rows in out.items():
  print(k,[(r['name'],round(r['volume_mm3'],1)) for r in rows],flush=True)
finally:
 pythoncom.CoUninitialize()
