import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  m=c.GetModelDoc2
  print(c.Name2, 'model',bool(m), 'type',m.GetType if m else None)
  if m:
   try: print('bodies',len(m.GetBodies2(0,True)))
   except Exception as e: print('body error',str(e))
