from pathlib import Path
import trimesh
p=Path(r'C:\Practice_project\biped_sim');f=next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL'));m=trimesh.load_mesh(f,process=True);out=p/'chassis_collision_pieces';out.mkdir(exist_ok=True);pieces=m.split(only_watertight=False,repair=False);n=0
for c in pieces:
 if not c.is_watertight or len(c.faces)<20:continue
 try:
  h=c.convex_hull
  if h.volume<.0001:continue
  h.export(out/f'chassis_{n}.stl'); n+=1
 except Exception:continue
print('watertight chassis convex pieces',n)
