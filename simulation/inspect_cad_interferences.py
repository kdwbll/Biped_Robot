import win32com.client as w, pythoncom
s=w.Dispatch('SldWorks.Application.33');err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
d=s.OpenDoc6(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM',2,3,'',err,warn)
print('opened',bool(d),'error',err.value,'warning',warn.value,flush=True)
if d:
 mgr=d.InterferenceDetectionManager
 try:
  mgr.TreatCoincidenceAsInterference=False
  print('calculate interference',flush=True); entries=mgr.GetInterferences
  print('interference count',0 if entries is None else len(entries),flush=True)
  if entries:
   for item in entries[:30]:
    try: print('components',[c.Name2 for c in item.Components],'volume',item.Volume,flush=True)
    except Exception as e: print('details_error',str(e),flush=True)
 finally:
  try:mgr.Done()
  except Exception:pass

