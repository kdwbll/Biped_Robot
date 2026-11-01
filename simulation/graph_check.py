import json
j=json.load(open(r'C:\Practice_project\biped_sim\axis_candidates.json',encoding='utf8'))
for name,axes in j.items():
 print('\n',name)
 for a in axes: print(' - '.join(a['components']), [round(a['axis_point_m'][i]*1000,1) for i in (0,2)])
