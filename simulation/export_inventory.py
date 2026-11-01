import json
import win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
out={'document':d.GetPathName,'components':[]}
for c in d.GetComponents(False):
 item={'name':c.Name2,'path':c.GetPathName,'suppressed':bool(c.IsSuppressed)}
 try:
  t=c.Transform2
  item['transform']=list(t.ArrayData) if t else None
 except Exception as e: item['transform_error']=str(e)
 out['components'].append(item)
with open(r'C:\Practice_project\biped_sim\assembly_inventory.json','w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
print('Saved component inventory:',len(out['components']))
