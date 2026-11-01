import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc;print('active',d.GetTitle);sel=d.SelectionManager;print('count',sel.GetSelectedObjectCount2(-1))
for i in range(1,sel.GetSelectedObjectCount2(-1)+1):
 print('type',sel.GetSelectedObjectType3(i,-1));o=sel.GetSelectedObject6(i,-1)
 try:
  curve=o.GetCurve; print('curve circle',curve.IsCircle, 'params',curve.CircleParams)
 except Exception as e: print('error',str(e))

