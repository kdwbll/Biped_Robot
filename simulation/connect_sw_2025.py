import win32com.client as w
s=w.Dispatch('SldWorks.Application.33');print('rev',s.RevisionNumber(),flush=True)
