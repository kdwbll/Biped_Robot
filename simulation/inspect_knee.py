import json
p=r'C:\Practice_project\biped_sim\axis_candidates.json'
j=json.load(open(p,encoding='utf8'))
for k,axes in j.items():
 print('\n',k)
 for a in axes:
  if any('18-52-233' in c for c in a['components']): print(a['mate'],a['components'],a['axis_point_m'])
