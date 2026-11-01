import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
for c in d.GetComponents(False):
 if '16-17-29-589' not in c.Name2 or c.IsSuppressed:continue
 print('\n',c.Name2)
 t=c.Transform2.ArrayData
 for b in c.GetModelDoc2.GetBodies2(0,True):
  try:
   box=b.GetBodyBox(); print('body box',tuple(round(x,4) for x in box))
  except Exception as e: print(e)
