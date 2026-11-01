import json,numpy as np,win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc;sel=d.SelectionManager;c=sel.GetSelectedObjectsComponent4(1,-1);edge=sel.GetSelectedObject6(1,-1); circle=edge.GetCurve.CircleParams;t=c.Transform2.ArrayData
p=np.array(circle[:3]); R=np.array(t[:9]).reshape(3,3); global_point=p@R + np.array(t[9:12]);print('selected edge center in assembly',global_point)
axes=list(json.load(open(r'C:\Practice_project\biped_sim\axis_candidates.json',encoding='utf8')).values())[0];matches=[]
for a in axes:
 if c.Name2 not in a['components']: continue
 u=np.array(a['axis_direction']);q=np.array(a['axis_point_m']); delta=global_point-q; distance=np.linalg.norm(delta-u*np.dot(delta,u));matches.append((distance,a['mate'],a['components']))
for dist,m,comps in sorted(matches):print('axis distance mm',dist*1000,'mate',m,'components',comps)
