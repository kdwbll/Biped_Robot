"""Check whether candidate missing CAD solid occurs in the leg's existing visual mesh."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import trimesh

p = Path(__file__).parent
out=[]
for leg, filename in enumerate(('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml')):
    root = ET.parse(p/filename).getroot()
    mesh_path = p / root.find('./asset/mesh').get('file')
    mesh = trimesh.load_mesh(mesh_path, process=True)
    parts = mesh.split(only_watertight=False,repair=False)
    matches = [{'extent_mm':c.extents.tolist(),'volume_mm3':float(c.volume),'watertight':bool(c.is_watertight)}
               for c in parts if np.allclose(sorted(c.extents),[19,85,85],atol=.2)]
    out.append({'leg':leg,'mesh':str(mesh_path),'component_count':len(parts),'matches':matches})
    print('leg',leg,'mesh',mesh_path.name,'candidate 19x85x85 matches',len(matches))
(p/'missing_solid_mesh_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
