import json
j=json.load(open(r'C:\Practice_project\biped_sim\assembly_inventory.json',encoding='utf8'))
for c in j['components']:
 if c['name'].count('/')==1 and '32-05-068' in c['name']:
  print(c['name'],c['transform'])
