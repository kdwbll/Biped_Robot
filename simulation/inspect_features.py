import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
f=d.FirstFeature
while f:
 print(f.Name, f.GetTypeName2)
 f=f.GetNextFeature

