import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
f=d.FirstFeature
while f:
 if f.GetTypeName2 == 'MateGroup':
  sub=f.GetFirstSubFeature
  while sub:
   print(sub.Name, sub.GetTypeName2)
   sub=sub.GetNextSubFeature
 f=f.GetNextFeature
