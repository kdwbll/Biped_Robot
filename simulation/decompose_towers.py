from pathlib import Path
import trimesh,coacd,xml.etree.ElementTree as E
p=Path(r'C:\Practice_project\biped_sim');coacd.set_log_level('error');out=p/'tower_collision_pieces';out.mkdir(exist_ok=True)
for leg in (0,1):
 root=E.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')).getroot();f=p/root.find("./asset/mesh[@name='mesh0']").get('file');m=trimesh.load_mesh(f,process=True);parts=m.split(only_watertight=False,repair=False); tower=max((x for x in parts if x.is_watertight and x.extents[2]>100),key=lambda a: len(a.faces));print('start',leg,'faces',len(tower.faces),flush=True)
 meshes=coacd.run_coacd(coacd.Mesh(tower.vertices.astype('float64'),tower.faces.astype('int32')),threshold=.08,max_convex_hull=18,resolution=700,mcts_iterations=40,preprocess_mode='off')
 for i,(v,f) in enumerate(meshes):trimesh.Trimesh(v,f,process=False).export(out/f'L{leg}_tower_{i}.stl')
 print('done',leg,len(meshes),flush=True)
