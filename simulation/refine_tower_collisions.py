"""Generate experimental finer convex approximations of one tower STL.

This does not modify the existing collision model and is not training.
"""
from pathlib import Path
import json
import sys
import time
import xml.etree.ElementTree as ET
import trimesh
import coacd

p=Path(__file__).parent
leg=int(sys.argv[1])
if leg not in (0,1):raise ValueError('Expected leg 0 or 1')
coacd.set_log_level('error')
root=ET.parse(p/('single_leg_mesh_experiment.xml' if leg==0 else 'second_leg_mesh_experiment.xml')).getroot()
path=p/root.find("./asset/mesh[@name='mesh0']").get('file')
mesh=trimesh.load_mesh(path,process=True)
parts=mesh.split(only_watertight=False,repair=False)
tower=max((x for x in parts if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces))
out=p/'tower_collision_refined';out.mkdir(exist_ok=True)
print('leg',leg,'input faces',len(tower.faces),'starting refined convex decomposition',flush=True)
started=time.monotonic()
pieces=coacd.run_coacd(coacd.Mesh(tower.vertices.astype('float64'),tower.faces.astype('int32')),
                       threshold=.025,max_convex_hull=32,resolution=500,mcts_iterations=30,preprocess_mode='off')
for i,(vertices,faces) in enumerate(pieces):
 trimesh.Trimesh(vertices,faces,process=False).export(out/f'L{leg}_tower_{i}.stl')
summary={'leg':leg,'source':str(path),'input_faces':len(tower.faces),'threshold':.025,
         'max_convex_hull':32,'resolution':500,'mcts_iterations':30,'output_count':len(pieces),
         'seconds':time.monotonic()-started}
(out/f'L{leg}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('finished leg',leg,'pieces',len(pieces),'seconds',round(summary['seconds'],1),flush=True)
