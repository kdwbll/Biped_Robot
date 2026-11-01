import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc;sel=d.SelectionManager;comp=sel.GetSelectedObjectsComponent4(1,-1);print('component',comp.Name2,'transform',comp.Transform2.ArrayData);edge=sel.GetSelectedObject6(1,-1);print('edge circle local',edge.GetCurve.CircleParams)
