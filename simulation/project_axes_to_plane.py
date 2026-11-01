import json
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim\axis_candidates.json');a=json.loads(p.read_text(encoding='utf8'))
for rows in a.values():
 for row in rows:
  q=row['axis_point_m']; u=row['axis_direction']; k=q[1]/u[1]; row['axis_point_m']=[q[i]-k*u[i] for i in range(3)]
p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf8');print('Projected axis points to leg modelling plane, preserving each 3D axis line.')
