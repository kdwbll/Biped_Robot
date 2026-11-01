import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); c=next(c for c in s.ActiveDoc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed); b=c.GetModelDoc2.GetBodies2(0,True)[0]
for name in ('GetTessTriangles','GetBodyBox','GetTessTriangleCount'):
 try:
  fn=getattr(b,name); v=fn(True) if name=='GetTessTriangles' else fn(); print(name, str(v)[:450])
 except Exception as e: print(name,'ERROR',str(e))
