import win32com.client as w, numpy as np
s=w.GetActiveObject('SldWorks.Application.33');d=s.ActiveDoc
for c in d.GetComponents(False):
 if '16-17-29-589' in c.Name2:
  t=c.Transform2.ArrayData; print(c.Name2, 'center global m',np.array([.042,.032,.28])@np.array(t[:9]).reshape(3,3)+np.array(t[9:12]))
