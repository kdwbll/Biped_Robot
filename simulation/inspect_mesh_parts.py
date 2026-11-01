from pathlib import Path
import trimesh
p=Path(r'C:\Practice_project\biped_sim\meshes_original_coords')
for f in p.glob('*.STL'):
 m=trimesh.load_mesh(f,process=True);c=m.split(only_watertight=False,repair=False);print(f.name[-45:], 'parts',len(c),'faces',len(m.faces),'watertight',sum(x.is_watertight for x in c))

