import json
j=json.load(open(r'C:\Practice_project\biped_sim\axis_candidates.json',encoding='utf8')); a=list(j.values())[0]; parts=sorted({v for x in a for v in x['components']})
for i,p in enumerate(parts): print('part'+str(i),p)
for x in a[:2]: print('base connection',x['components'])
