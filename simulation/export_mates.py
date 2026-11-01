import json,win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
out=[]
for c in d.GetComponents(False):
 if not c.GetPathName.lower().endswith('.sldasm'): continue
 m=c.GetModelDoc2
 if not m: continue
 f=m.FirstFeature
 while f:
  if f.GetTypeName2=='MateGroup':
   sub=f.GetFirstSubFeature
   while sub:
    mate=sub.GetSpecificFeature2
    row={'assembly':c.Name2,'name':sub.Name,'type':sub.GetTypeName2,'entities':[]}
    try:
     for i in range(mate.GetMateEntityCount):
      e=mate.MateEntity(i); ref=e.ReferenceComponent
      row['entities'].append({'component':ref.Name2 if ref else None,'parameters':list(e.EntityParams)})
    except Exception as ex: row['error']=str(ex)
    out.append(row); sub=sub.GetNextSubFeature
  f=f.GetNextFeature
with open(r'C:\Practice_project\biped_sim\mates.json','w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
print('Extracted',len(out),'mates,',sum(len(x['entities']) for x in out),'entities')
