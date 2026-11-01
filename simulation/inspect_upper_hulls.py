from pathlib import Path
import trimesh
p=Path(r'C:\Practice_project\biped_sim\upper_collision_pieces')
for f in p.glob('*.stl'):
 m=trimesh.load_mesh(f);print(f.name, m.extents.round(1))
