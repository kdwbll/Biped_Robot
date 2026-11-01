"""Save a conservative numerical gear-constraint candidate; no training."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
tree=ET.parse(p/'biped_fresh_mass_only_candidate.xml'); root=tree.getroot()
root.set('model','biped_standing_gear_candidate')
root.find('option').set('timestep','0.00025')
for eq in root.findall('./equality/joint'):
 if eq.get('name') in ('opposite_world_rotation_assumption','estimated_pinion_23_gear_50'):
  eq.set('solref','0.001 1')
  eq.set('solimp','0.99 0.999 0.001 0.5 2')
out=p/'biped_standing_gear_candidate.xml';ET.indent(root);tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out))
print('Created',out.name,'timestep',m.opt.timestep,'equalities',m.neq,'actuators',m.nu)
