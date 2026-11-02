"""Check restored pair selection and ensure all other XML content is unchanged."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
p=Path(__file__).parent
base=ET.parse(p/'diagnostic_separate_sole_contact_trial.xml').getroot();trial=ET.parse(p/'diagnostic_restored_near_pairs_trial.xml').getroot()
base.attrib.pop('model',None);trial.attrib.pop('model',None)
selected=json.loads((p/'near_collision_gap_refinement_audit.json').read_text(encoding='utf8'))['results'];expected={tuple(r['geoms']) for r in selected}
contact=trial.find('contact');found=set()
for pair in list(contact.findall('pair')):
 key=(pair.get('geom1'),pair.get('geom2'))
 if key in expected:
  if key in found:raise RuntimeError('Duplicate')
  found.add(key);contact.remove(pair)
if found!=expected or ET.tostring(base)!=ET.tostring(trial):raise RuntimeError('Unintended XML differences')
out={'verified':True,'restored_pair_count':len(found),'unchanged_other_xml':True,'limitations':'Configuration equivalence and sampled bounds only; does not prove physical collision correctness.'}
(p/'restored_near_pairs_difference_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out)
