"""Create an isolated candidate adding one CAD solid missed by old inventory."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

p = Path(__file__).parent
inventory = json.loads((p / 'cad_mass_inventory_current.json').read_text(encoding='utf8'))
matches = [r for r in inventory if r['body'] == 'L1_part0' and
           np.allclose(sorted(r['box_size_m']), [.019, .085, .085], atol=.0002)]
if len(matches) != 1:
    raise RuntimeError(f'Expected exactly one missing mirrored solid, found {len(matches)}')
entry = matches[0]
tree = ET.parse(p / 'biped_filtered_collision_experimental.xml')
root = tree.getroot()
inertial = root.find(".//body[@name='L1_part0']/inertial")
old_mass = float(inertial.get('mass'))
old_center = np.fromstring(inertial.get('pos'), sep=' ')
v = np.fromstring(inertial.get('fullinertia'), sep=' ')
old_i = np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
new_mass = entry['mass_kg']
new_center = np.array(entry['center_m'])
new_i = np.array(entry['approx_inertia_at_center'])
total = old_mass + new_mass
center = (old_mass*old_center + new_mass*new_center)/total
def shift(mass, displacement):
    return mass*(np.dot(displacement, displacement)*np.eye(3)-np.outer(displacement, displacement))
inertia = old_i + shift(old_mass, old_center-center) + new_i + shift(new_mass, new_center-center)
if np.linalg.eigvalsh(inertia).min() <= 0:
    raise RuntimeError('Mass tensor is invalid')
inertial.set('mass', f'{total:.12g}')
inertial.set('pos', ' '.join(f'{x:.12g}' for x in center))
inertial.set('fullinertia', ' '.join(f'{x:.12g}' for x in
             [inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
root.set('model', 'biped_fresh_mass_only_candidate')
out = p / 'biped_fresh_mass_only_candidate.xml'
ET.indent(root)
tree.write(out, encoding='utf8', xml_declaration=True)
old = mujoco.MjModel.from_xml_path(str(p/'biped_filtered_collision_experimental.xml'))
new = mujoco.MjModel.from_xml_path(str(out))
print('Added CAD solid:', entry['solid_name'], 'mass g', round(new_mass*1000, 4))
print('Old/new model total kg', round(mujoco.mj_getTotalmass(old), 6), round(mujoco.mj_getTotalmass(new), 6))
print('Only mass and inertia updated; collision/visual meshes, keyframe and other parameters unchanged.')
