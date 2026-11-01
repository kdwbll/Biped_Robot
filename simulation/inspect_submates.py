import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldasm'):
  m=c.GetModelDoc2
  print('\nASSEMBLY',c.Name2, bool(m))
  if not m: continue
  f=m.FirstFeature
  while f:
   if f.GetTypeName2 == 'MateGroup':
    sub=f.GetFirstSubFeature
    while sub:
     print('MATE',sub.Name,sub.GetTypeName2)
     sub=sub.GetNextSubFeature
   f=f.GetNextFeature
