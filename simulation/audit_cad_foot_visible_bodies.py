"""Read visible foot bodies in the open SolidWorks 2025 assembly without saving."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w
p=Path(__file__).parent
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved assembly in SolidWorks 2025')
 out=[]
 for comp in doc.GetComponents(False):
  if '16-19-39-145' not in comp.Name2 or comp.IsSuppressed:continue
  part=comp.GetModelDoc2
  if part is None:raise RuntimeError('Foot component not resolved')
  bodies=[]
  for body in part.GetBodies2(0,True) or []:
   bodies.append({'name':body.Name,'bounds_local_mm':[x*1000 for x in body.GetBodyBox()],'volume_mm3':body.GetMassProperties(1)[3]*1e9})
  out.append({'component':comp.Name2,'source':comp.GetPathName,'visible_bodies':bodies})
 if len(out)!=2:raise RuntimeError(f'Expected two visible feet, found {len(out)}')
 report={'feet':out,'limitations':'Reads current open CAD state without saving; compare with saved file separately if unsaved edits exist. Visible bodies only, not design or hardware certification.'}
 (p/'cad_foot_visible_bodies_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 for f in out: print(f['component'],len(f['visible_bodies']),[round(b['volume_mm3'],1) for b in f['visible_bodies']],flush=True)
finally:pythoncom.CoUninitialize()
