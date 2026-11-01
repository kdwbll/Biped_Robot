import os, pythoncom, win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
out=r'C:\Practice_project\biped_sim\meshes'; os.makedirs(out,exist_ok=True)
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  m=c.GetModelDoc2
  e=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0); warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
  path=os.path.join(out,os.path.basename(c.GetPathName).rsplit('.',1)[0]+'.stl')
  try:
   ok=m.Extension.SaveAs(path,0,3,None,e,warn)
   print(os.path.basename(path), 'ok',ok,'error',e.value,'warning',warn.value)
  except Exception as ex: print('ERROR',path,str(ex))
