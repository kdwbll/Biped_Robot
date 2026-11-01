"""Create a separate phase-adjusted visualization and kinematic diagnostic.

Gear phase offsets are CAD-body search candidates, not a claim of safe physical
limits. The ±5 degree hip range is a temporary hypothesis, not a real limit.
"""
from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET
import trimesh
import mujoco
import numpy as np

p=Path(__file__).parent
ratio25='--25-tooth' in sys.argv
ratio=.5 if ratio25 else .46
search=json.loads((p/'cad_gear_phase_search.json').read_text(encoding='utf8'))
small_deg=search['search']['small']['best']['angle_deg']
large_deg=search['search']['other_large']['best']['angle_deg']
root=ET.parse(p/'diagnostic_post_shaft_fix.xml').getroot()
asset=root.find('asset');chassis=asset.find("mesh[@name='chassis_mesh']")
original=p/chassis.get('file');mesh=trimesh.load_mesh(original,process=True)
parts=mesh.split(only_watertight=False,repair=False)
matches=[i for i,x in enumerate(parts) if 18900 < abs(x.volume) < 19500 and
         np.allclose(sorted(x.extents),[9.68,56.36,56.36],atol=.5)]
if len(matches)!=1:raise RuntimeError(f'Expected one pinion mesh, got {matches}')
idx=matches[0];pinion=parts[idx].copy();remaining=trimesh.util.concatenate([x for i,x in enumerate(parts) if i!=idx])
if len(pinion.faces)+len(remaining.faces)!=len(mesh.faces):raise RuntimeError('Lost faces during visual split')
outdir=p/'gear_phase_visuals';outdir.mkdir(exist_ok=True)
remaining.export(outdir/'chassis_without_pinion.stl')
shaft=root.find(".//body[@name='hypothetical_motor_shaft']")
center=np.fromstring(shaft.get('pos'),sep=' ')
pinion.apply_translation(-center*1000)
pinion.export(outdir/'pinion_local.stl')
chassis.set('file','gear_phase_visuals/chassis_without_pinion.stl')
ET.SubElement(asset,'mesh',name='phase_pinion_mesh',file='gear_phase_visuals/pinion_local.stl',scale='.001 .001 .001')
ET.SubElement(shaft,'geom',name='phase_pinion_visual',type='mesh',mesh='phase_pinion_mesh',
              contype='0',conaffinity='0',mass='0',rgba='0.6 0.65 0.72 1')
q=np.fromstring(root.find('./keyframe/key').get('qpos'),sep=' ')
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix.xml'))
names=['L0_hip_candidate','L1_hip_candidate','hypothetical_servo_shaft']
joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
q0,q1,qs=[int(m.jnt_qposadr[j]) for j in joints]
q[q1]-=math.radians(large_deg);q[qs]+=math.radians(small_deg)
# Keep equalities satisfied around the changed phase; coefficients are radians.
equal=root.find('equality')
offset1=q[q1]-q[q0]
offset2=q[q0]+ratio*q[qs]
equal.find("joint[@name='opposite_world_rotation_assumption']").set('polycoef',f'{offset1:.15g} 1 0 0 0')
gear_equal=equal.find("joint[@name='estimated_pinion_23_gear_50']")
gear_equal.set('polycoef',f'{offset2:.15g} {-ratio:.15g} 0 0 0')
if ratio25:gear_equal.set('name','candidate_pinion_25_gear_50')
for name in names[:2]:root.find(f".//joint[@name='{name}']").set('range','-5 5')
root.find('./keyframe/key').set('qpos',' '.join(f'{v:.15g}' for v in q))
root.set('model','diagnostic_post_shaft_fix_gear_phase_25' if ratio25 else 'diagnostic_post_shaft_fix_gear_phase');ET.indent(root)
out=p/('diagnostic_post_shaft_fix_gear_phase_25.xml' if ratio25 else 'diagnostic_post_shaft_fix_gear_phase.xml');ET.ElementTree(root).write(out,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(out));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
errs=[]
for name in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50' if ratio25 else 'estimated_pinion_23_gear_50'):
 eq=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,name)
 match=np.flatnonzero((data.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id==eq))
 errs.append(float(data.efc_pos[match[0]]))
summary={'small_gear_world_z_phase_deg':small_deg,'other_large_gear_world_z_phase_deg':large_deg,'pinion_to_large_ratio':ratio,
         'hip_range_degrees_assumption':[-5,5],'baseline_gear_constraint_residual_rad':errs,
         'original_chassis_faces':len(mesh.faces),'split_chassis_faces':len(pinion.faces)+len(remaining.faces),
         'note':'Visualization pinion now rotates with shaft; gear masses and collision proxies are unchanged. CAD remains untouched.'}
(p/('gear_phase_25_diagnostic_build.json' if ratio25 else 'gear_phase_diagnostic_build.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
print('small phase deg',round(small_deg,4),'other large phase deg',round(large_deg,4),'eq residual',errs)
print('WARNING: assumed hip range +/-5 degrees; collision coverage and gear masses unchanged')
