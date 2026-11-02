"""Verify intended sector-pair replacement, no unintended non-contact changes."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
p=Path(__file__).parent
base=ET.parse(p/'diagnostic_sector_gap_only.xml').getroot();trial=ET.parse(p/'diagnostic_sector_collision_trial.xml').getroot()
sectors=json.loads((p/'sector_clip_prototype_audit.json').read_text())['results'];mapping={f"nomerge_{x['geom']}":[f"sector_{Path(f).stem}" for f in x['files']] for x in sectors}
safe=[x for x in json.loads((p/'sector_gap_only_sweep_audit.json').read_text())['results'] if x['bounded_lower_mm']>.05]
def keys(root):
 pairs=[(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair')]
 if len(set(pairs))!=len(pairs):raise RuntimeError('Duplicate ordered pair')
 return set(pairs)
expected=set()
for a,b in keys(base):
 if a in mapping:expected.update((g,b) for g in mapping[a])
 elif b in mapping:expected.update((a,g) for g in mapping[b])
 else:expected.add((a,b))
for row in safe:
 a,b=row['geoms'];expected.update((g,b) for g in mapping[a])
actual=keys(trial)
if len({frozenset(x) for x in actual})!=len(actual):raise RuntimeError('Duplicate collision pair in either orientation')
if expected!=actual:raise RuntimeError(('pair mismatch',len(expected-actual),len(actual-expected)))
trial_by_key={(x.get('geom1'),x.get('geom2')):x for x in trial.find('contact').findall('pair')}
for old in base.find('contact').findall('pair'):
 a,b=old.get('geom1'),old.get('geom2');source=a if a in mapping else b if b in mapping else None
 targets=[(g,b) if source==a else (a,g) for g in mapping[source]] if source else [(a,b)]
 for key in targets:
  actual_attr={k:v for k,v in trial_by_key[key].attrib.items() if k not in ('geom1','geom2','name')}
  expected_attr={k:v for k,v in old.attrib.items() if k not in ('geom1','geom2','name')}
  if actual_attr!=expected_attr:raise RuntimeError(('pair parameters changed',key))
for row in safe:
 a,b=row['geoms']
 for g in mapping[a]:
  params=trial_by_key[(g,b)].attrib
  if params.get('solref')!='.001 1' or params.get('solimp')!='.99 .999 .001':raise RuntimeError('Added pair parameters wrong')
base.attrib.pop('model',None);trial.attrib.pop('model',None);base.remove(base.find('contact'));trial.remove(trial.find('contact'))
if ET.tostring(base)!=ET.tostring(trial):raise RuntimeError('Non-contact XML changed')
out={'verified':True,'expected_pairs':len(expected),'safe_original_pairs_added':len(safe),'noncontact_xml_unchanged':True,'limitations':'Checks pair identities and attributes, not CAD fidelity or dynamic correctness.'}
(p/'sector_collision_selection_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
