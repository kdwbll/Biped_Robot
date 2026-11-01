from pathlib import Path
import trimesh,xml.etree.ElementTree as E
p=Path(r'C:\Practice_project\biped_sim');out=p/'upper_collision_pieces';out.mkdir(exist_ok=True)
for leg in (0,1):
 root=E.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')).getroot();f=p/root.find("./asset/mesh[@name='mesh0']").get('file');mesh=trimesh.load_mesh(f,process=True);parts=mesh.split(only_watertight=False,repair=False);n=0
 for part in parts:
  if not part.is_watertight or len(part.faces)<300:continue
  try:
   hull=part.convex_hull
   if hull.volume<100:continue
   hull.export(out/f'L{leg}_part0_{n}.stl');n+=1
  except Exception:pass
 print('leg',leg,'upper convex pieces',n)
