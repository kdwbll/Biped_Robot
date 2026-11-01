import json
p=r'C:\Practice_project\biped_sim\mates.json'
a=json.load(open(p,encoding='utf8'))
for x in a:
 if x['type']=='MateConcentric':
  print(x['assembly'].split('/')[-1],x['name'],[(e['component'],[round(y,4) for y in e['parameters'][:6]]) for e in x['entities']])
