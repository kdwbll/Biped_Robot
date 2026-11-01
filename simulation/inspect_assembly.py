import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
print('PATH', d.GetPathName)
a=w.Dispatch(d)
components=a.GetComponents(False)
print('COUNT',len(components))
for c in components:
 try:
  print(c.Name2, '|', c.GetPathName, '| suppressed:', c.IsSuppressed)
 except Exception as e: print('error',str(e))

