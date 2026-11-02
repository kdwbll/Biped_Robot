"""Measure proxy-vs-tower distance for sector-clipped diagnostic meshes."""
from pathlib import Path
import json
import mujoco
import numpy as np
p=Path(__file__).parent
model=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_gap_only.xml'));d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,0);mujoco.mj_forward(model,d)
sectors=json.loads((p/'sector_clip_prototype_audit.json').read_text(encoding='utf8'))['results'];mapping={f"nomerge_{x['geom']}":[f"sector_{Path(f).stem}" for f in x['files']] for x in sectors}
source=json.loads((p/'chassis_witness_stl_containment_audit.json').read_text(encoding='utf8'))['results'];rows=[]
for x in source:
 if x['cad_nearest_surface_distance_mm']<=.1 or x['stl_inside'] or x['geoms'][0] not in mapping:continue
 center=np.array([42.,32. if x['leg']==0 else -73.]);r=np.linalg.norm(np.array(x['point_assembly_mm'][:2])-center)
 if not 19.5<=r<=20.3:continue
 a,b=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in x['geoms']]
 old=float(mujoco.mj_geomDistance(model,d,a,b,.003,None)*1000)
 distances=[float(mujoco.mj_geomDistance(model,d,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n),b,.003,None)*1000) for n in mapping[x['geoms'][0]]]
 rows.append({'geoms':x['geoms'],'leg':x['leg'],'old_gap_mm':old,'sector_gap_mm':min(distances),'piece_count':len(distances)})
if len(rows)!=57:raise RuntimeError(f'Unexpected sample count {len(rows)}')
out={'sample_count':len(rows),'old_min_mm':min(x['old_gap_mm'] for x in rows),'new_min_mm':min(x['sector_gap_mm'] for x in rows),'new_max_mm':max(x['sector_gap_mm'] for x in rows),'results':rows,'limitations':'Initial keyframe only; compares selected earlier positive-gap proxy pairs, not all 790 or a full sweep. Sector geometry is inactive and no contact dynamics or seams have been tested. Separation capped at 3 mm.'}
(p/'sector_gap_only_witness_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('samples',len(rows),'min old/new mm',out['old_min_mm'],out['new_min_mm'],'max new mm',out['new_max_mm'],flush=True)
