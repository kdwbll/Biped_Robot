import json
j=json.load(open(r'C:\Practice_project\biped_sim\axis_candidates.json',encoding='utf8'))
for i,axes in enumerate(j.values()):
 parts=sorted({v for x in axes for v in x['components']});print(i,[(n,p[11:]) for n,p in enumerate(parts)])
