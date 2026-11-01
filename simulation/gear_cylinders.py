import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc
c=next(c for c in d.GetComponents(False) if c.Name2.endswith('16-17-29-589-1'))
for i,b in enumerate(c.GetModelDoc2.GetBodies2(0,True)):
 box=b.GetBodyBox()
 if box[2] < 0.26:continue
 print('\nBODY',i,tuple(round(x,4) for x in box))
 for face in b.GetFaces():
  try:
   surf=face.GetSurface
   if surf.IsCylinder:
    p=surf.CylinderParams
    if abs(p[5])>.99:print('cylinder',tuple(round(x,5) for x in p))
  except Exception as e: print('error',str(e));break
