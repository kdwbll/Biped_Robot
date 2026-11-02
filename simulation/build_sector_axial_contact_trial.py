"""Create diagnostic candidate without one axial redundant proxy contact."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent
bounds=json.loads((p/'peak_pair_axial_bounds_audit.json').read_text());peak=json.loads((p/'sector_peak_contact_direction_audit.json').read_text())['peak'];cad=json.loads((p/'cad_peak_force_pair_sweep.json').read_text())
if abs(bounds['axial_gap_mm'])>.001 or peak['normal_axis_abs_dot']<.999 or peak['normal_relative_jacobian_max_abs']>1e-7 or any(x['intersects'] for x in cad['results']):raise RuntimeError('Insufficient evidence for diagnostic candidate')
tree=ET.parse(p/'diagnostic_sector_collision_trial.xml');root=tree.getroot();contact=root.find('contact');target={'dc_chassis_1_1','tower_L0_tower_8'}
pairs=[x for x in contact.findall('pair') if {x.get('geom1'),x.get('geom2')}==target]
if len(pairs)!=1:raise RuntimeError('Expected exactly one explicit pair')
attrs=dict(pairs[0].attrib);contact.remove(pairs[0]);root.set('model','diagnostic_sector_axial_contact_trial');out=p/'diagnostic_sector_axial_contact_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));report={'removed_pair':attrs,'compiled_pair_count':int(m.npair),'limitations':'Removes only one explicit contact under ideal rigid hinge assumptions. Does not model actual normal loading or friction; other contact omissions and real clearances remain. No XML range guard; diagnostic only.'}
(p/'sector_axial_contact_trial_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
