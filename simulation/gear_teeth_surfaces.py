import win32com.client as w,collections
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc
for c in d.GetComponents(False):
 if not (c.Name2.endswith('16-40-03-958-1') or '16-17-29-589' in c.Name2):continue
 print('\n',c.Name2)
 for i,b in enumerate(c.GetModelDoc2.GetBodies2(0,True)):
  count=collections.Counter()
  for f in b.GetFaces():
   try:
    surf=f.GetSurface
    if surf.IsCylinder:
     q=surf.CylinderParams
     if abs(q[5])>.99: count[(round(q[0],5),round(q[1],5),round(q[6],5))]+=1
   except:pass
  for k,v in count.items():
   if v>=10:print('body',i,'axis radius',k,'face count',v)
