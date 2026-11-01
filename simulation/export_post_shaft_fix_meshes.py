"""Export saved assembly STL to a separate directory, preserving SW settings."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

p=Path(__file__).parent
source=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
out=p/'meshes_after_shaft_fix';out.mkdir(exist_ok=True)
pythoncom.CoInitialize();sw=w.DispatchEx('SldWorks.Application.33');sw.Visible=False;doc=None
setting=None
try:
 err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 doc=sw.OpenDoc6(source,2,3,'',err,warn)
 if not doc:raise RuntimeError((err.value,warn.value))
 setting=sw.GetUserPreferenceToggle(71)
 sw.SetUserPreferenceToggle(71,True)
 if not sw.GetUserPreferenceToggle(71):raise RuntimeError('Cannot enable per-component STL export')
 status=doc.SaveAs3(str(out/'assembly.stl'),0,2)
 files=sorted(f.name for f in out.glob('*.stl'))
 result={'source':source,'requested_readonly':True,'open_errors':err.value,'open_warnings':warn.value,
         'saveas_status':status,'files':files,'old_setting':bool(setting)}
 (out/'export_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print('status',status,'STL count',len(files),'prior preference',setting,flush=True)
 if not files:raise RuntimeError('No STL files were exported')
finally:
 if setting is not None:sw.SetUserPreferenceToggle(71,setting)
 if doc:sw.CloseDoc(doc.GetTitle)
 sw.ExitApp();pythoncom.CoUninitialize()
