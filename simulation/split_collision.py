from pathlib import Path
import trimesh
p=Path(r'C:\Practice_project\biped_sim'); out=p/'collision_pieces';out.mkdir(exist_ok=True)
for leg in (0,1):
 for j in (1,2,4,5,6):
  import xml.etree.ElementTree as E
  root=E.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')).getroot()
  f=p/root.find(f"./asset/mesh[@name='mesh{j}']").get('file');mesh=trimesh.load_mesh(f,process=True);pieces=mesh.split(only_watertight=False,repair=False);n=0
  for piece in pieces:
   if len(piece.faces)<20:continue
   try:
    hull=piece.convex_hull
    if hull.volume<=1e-4:continue
    hull.export(out/f'L{leg}_part{j}_{n}.stl');n+=1
   except Exception:continue
  print('leg',leg,'part',j,'convex pieces',n)
