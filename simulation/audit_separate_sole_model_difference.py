"""Assert the separated-foot trial changes only intended ground collision elements."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
p=Path(__file__).parent
a=ET.parse(p/'diagnostic_foot_suppressed_plate_trial.xml').getroot()
b=ET.parse(p/'diagnostic_separate_sole_contact_trial.xml').getroot()
a.attrib.pop('model',None);b.attrib.pop('model',None)
for leg in (0,1):
 for suffix in ('sole','foot_upper'):
  body=b.find(f".//body[@name='L{leg}_part6']")
  geom=body.find(f"geom[@name='L{leg}_{suffix}_trial_geom']")
  if geom is None:raise RuntimeError('missing new geometry')
  body.remove(geom)
  asset=b.find(f"./asset/mesh[@name='L{leg}_{suffix}_trial_mesh']")
  if asset is None:raise RuntimeError('missing new mesh')
  b.find('asset').remove(asset)
 pair=b.find(f"./contact/pair[@name='floor_L{leg}_6']")
 if pair.get('geom2')!=f'L{leg}_sole_trial_geom':raise RuntimeError('floor pair wrong')
 pair.set('geom2',f'L{leg}_visual6')
 upper=b.find(f"./contact/pair[@name='floor_L{leg}_upper_trial']")
 if upper is None or upper.get('geom2')!=f'L{leg}_foot_upper_trial_geom':raise RuntimeError('upper pair wrong')
 b.find('contact').remove(upper)
if ET.tostring(a)!=ET.tostring(b):raise RuntimeError('Unexpected difference outside intended ground contacts')
out={'result':'Only model name, four foot collision meshes/geometries and ground contact pairs differ; other XML including inertials, foot self-collision and drive unchanged','limitations':'XML equivalence is not physical validation; foot parts still convexified and collisions elsewhere incomplete.'}
(p/'separate_sole_model_difference_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out['result'])
