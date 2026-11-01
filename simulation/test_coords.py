import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  print(c.Name2, 'T',c.Transform2.ArrayData)
  m=c.GetModelDoc2
  print('BODY BOX', m.GetBodies2(0,True)[0].GetBodyBox)
  break
