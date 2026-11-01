import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc
for c in d.GetComponents(False):
 if '16-17-29-589' in c.Name2:
  print(c.Name2,c.Transform2.ArrayData)
  try: print('total', c.GetTotalTransform(False).ArrayData)
  except Exception as e: print(e)
