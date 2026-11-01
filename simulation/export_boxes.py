import json,win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
items=[]
for c in d.GetComponents(False):
 if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed:
  items.append({'name':c.Name2.split('/')[-1], 'box_m':list(c.GetBox(False,False))})
json.dump(items,open(r'C:\Practice_project\biped_sim\sw_boxes.json','w',encoding='utf8'),indent=2)
