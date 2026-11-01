import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); c=next(c for c in s.ActiveDoc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed); m=c.GetModelDoc2
try: print('SaveAs3 result',m.SaveAs3(r'C:\Practice_project\biped_sim\meshes\test.stl',0,2))
except Exception as e: print('error',str(e))
