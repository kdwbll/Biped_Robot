import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  try: print(c.Name2.split('/')[-1],tuple(round(x,4) for x in c.GetBox(False,False)))
  except Exception as ex: print('ERR',str(ex))
