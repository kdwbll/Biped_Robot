from pathlib import Path
import trimesh,coacd
p=Path(r'C:\Practice_project\biped_sim');coacd.set_log_level('error');f=next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL'));m=trimesh.load_mesh(f,process=True);out=p/'chassis_decomposed';out.mkdir(exist_ok=True);n=0
for c in m.split(only_watertight=False,repair=False):
 if not c.is_watertight or len(c.faces)<20:continue
 try:
  if c.convex_hull.volume<.0001:continue
  print('piece',n,'faces',len(c.faces),flush=True)
  parts=coacd.run_coacd(coacd.Mesh(c.vertices.astype('float64'),c.faces.astype('int32')),threshold=.08,max_convex_hull=8,resolution=400,mcts_iterations=20,preprocess_mode='off')
  for j,(v,faces) in enumerate(parts):trimesh.Trimesh(v,faces,process=False).export(out/f'chassis_{n}_{j}.stl')
  print('result',n,len(parts),flush=True);n+=1
 except Exception as e:print('FAILED',n,str(e),flush=True);raise
