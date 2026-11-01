import coacd,trimesh,xml.etree.ElementTree as E
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim');coacd.set_log_level('error')
for leg in (0,1):
 root=E.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')).getroot();f=p/root.find("./asset/mesh[@name='mesh3']").get('file');m=trimesh.load_mesh(f,process=True);print('start',leg,len(m.faces),flush=True)
 meshes=coacd.run_coacd(coacd.Mesh(m.vertices.astype('float64'),m.faces.astype('int32')),threshold=.12,max_convex_hull=12,preprocess_resolution=40,resolution=500,mcts_iterations=40)
 out=p/'knee_collision_pieces';out.mkdir(exist_ok=True)
 for i,(v,f) in enumerate(meshes):trimesh.Trimesh(v,f,process=False).export(out/f'L{leg}_part3_{i}.stl')
 print('completed',leg,len(meshes),flush=True)
