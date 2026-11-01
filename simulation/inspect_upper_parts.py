from pathlib import Path
import trimesh
p=Path(r'C:\Practice_project\biped_sim');mesh=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-17-29-589-1.STL')),process=True)
for i,c in enumerate(mesh.split(only_watertight=False,repair=False)):
 if len(c.faces)<30:continue
 print(i,'faces',len(c.faces),'watertight',c.is_watertight,'size mm',c.extents.round(2),'bounds center',c.bounds.mean(axis=0).round(2))
