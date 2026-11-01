"""Open the saved assembly read-only and list component/mate metadata.

This does not change or save the user's SolidWorks documents.
"""
from pathlib import Path
import json,traceback
import pythoncom,win32com.client as w

root=Path(__file__).parent
path=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
pythoncom.CoInitialize()
sw=w.DispatchEx('SldWorks.Application.33');sw.Visible=False
doc=None
try:
 err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 doc=sw.OpenDoc6(path,2,3,'',err,warn) # assembly; silent + read-only
 print('open success',bool(doc),'errors',err.value,'warnings',warn.value,flush=True)
 if not doc:raise RuntimeError('SolidWorks could not open the assembly')
 print('document',doc.GetPathName,'opened with silent + read-only options',flush=True)
 components=[]
 for c in doc.GetComponents(False):
  try:
   components.append(dict(name=c.Name2,path=c.GetPathName,suppressed=bool(c.IsSuppressed),
    loaded=bool(c.GetModelDoc2),transform=list(c.Transform2.ArrayData) if c.Transform2 else None))
  except Exception as exc:components.append(dict(name=str(c.Name2),error=str(exc)))
 print('component count',len(components),'suppressed',sum(c.get('suppressed',False) for c in components),
       'unresolved',sum(not c.get('loaded',False) and not c.get('suppressed',False) for c in components),flush=True)
 result=dict(document=doc.GetPathName,requested_open_options=3,open_errors=err.value,open_warnings=warn.value,components=components)
 (root/'sw_readonly_current_inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
finally:
 if doc:sw.CloseDoc(doc.GetTitle)
 sw.ExitApp()
 pythoncom.CoUninitialize()
