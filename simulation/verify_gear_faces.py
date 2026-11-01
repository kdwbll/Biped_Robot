import win32com.client as w, numpy as np
from collections import Counter
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc
for c in d.GetComponents(False):
 if not (c.Name2.endswith('16-40-03-958-1') or c.Name2.endswith('16-17-29-589-1')):continue
 for i,b in enumerate(c.GetModelDoc2.GetBodies2(0,True)):
  if (c.Name2.endswith('16-40-03-958-1') and i!=8) or (c.Name2.endswith('16-17-29-589-1') and i!=0):continue
  counter=Counter()
  for f in b.GetFaces():
   sf=f.GetSurface
   if sf.IsCylinder:
    q=sf.CylinderParams
    if abs(q[5])>.99:counter[(round(q[0],5),round(q[1],5),round(q[6],5))]+=1
  print(c.Name2,[(k,v) for k,v in counter.items() if v>=10])
