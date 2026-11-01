import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');
for d in s.GetDocuments:
 print('doc',d.GetTitle)
 if d.GetType==2:
  for c in d.GetComponents(False):
   if c.Name2.endswith('16-18-52-233-1'):
    print('component',c.Name2,'transform',c.Transform2.ArrayData,'box',c.GetBox(False,False))
