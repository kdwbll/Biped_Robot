import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33');d=next(x for x in s.GetDocuments if x.GetTitle=='SimpleBiped2a_new.SLDASM')
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  print('\n',c.Name2.split('/')[-1]); m=c.GetModelDoc2
  for b in m.GetBodies2(0,True):
   try:
    box=b.GetBodyBox();prop=b.GetMassProperties(1);print('size mm',tuple(round((box[i+3]-box[i])*1000,1) for i in range(3)), 'volume mm3',round(prop[3]*1e9,1), 'center',tuple(round(x*1000,1) for x in prop[:3]))
   except Exception as e:print('ERROR',str(e));break
