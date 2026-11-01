import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
c=next(c for c in d.GetComponents(False) if c.Name2.endswith('16-32-05-068-1'))
f=c.GetModelDoc2.FirstFeature
while f and f.GetTypeName2 != 'MateGroup': f=f.GetNextFeature
sub=f.GetFirstSubFeature
while sub:
 mate=sub.GetSpecificFeature2
 try:
  print(sub.Name,sub.GetTypeName2,'entities',mate.GetMateEntityCount)
  for i in range(mate.GetMateEntityCount):
   ent=mate.MateEntity(i)
   print(' ',ent.ReferenceComponent.Name2 if ent.ReferenceComponent else None, 'params',ent.EntityParams)
 except Exception as e: print('ERR',str(e))
 sub=sub.GetNextSubFeature
