"""Verify only the specified explicit contact differs in axial-contact trial."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
p=Path(__file__).parent;base=ET.parse(p/'diagnostic_sector_collision_trial.xml').getroot();trial=ET.parse(p/'diagnostic_sector_axial_contact_trial.xml').getroot()
base.attrib.pop('model',None);trial.attrib.pop('model',None);contact=base.find('contact');target={'dc_chassis_1_1','tower_L0_tower_8'}
found=[e for e in contact.findall('pair') if {e.get('geom1'),e.get('geom2')}==target]
if len(found)!=1:raise RuntimeError('Missing baseline pair')
contact.remove(found[0])
if ET.tostring(base)!=ET.tostring(trial):raise RuntimeError('Unexpected XML difference')
out={'single_contact_removed':True,'all_other_xml_unchanged':True,'limitations':'XML check only. The removed contact could carry real tangential friction or load not represented by ideal hinge.'}
(p/'sector_axial_contact_difference_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out)
