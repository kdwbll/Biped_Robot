"""Create sector-collision candidate; retain exclusions for unresolved pairs."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent
sectors=json.loads((p/'sector_clip_prototype_audit.json').read_text(encoding='utf8'))['results'];mapping={f"nomerge_{x['geom']}":[f"sector_{Path(f).stem}" for f in x['files']] for x in sectors}
sweep=json.loads((p/'sector_gap_only_sweep_audit.json').read_text(encoding='utf8'))['results'];safe=[x for x in sweep if x['bounded_lower_mm']>.05]
if len(mapping)!=18 or len(safe)!=56:raise RuntimeError('Unexpected sector audit')
tree=ET.parse(p/'diagnostic_sector_gap_only.xml');root=tree.getroot();contact=root.find('contact');replaced=[]
for pair in list(contact.findall('pair')):
 a,b=pair.get('geom1'),pair.get('geom2')
 if a in mapping and b in mapping:raise RuntimeError('Both geoms targeted')
 if a not in mapping and b not in mapping:continue
 old=a if a in mapping else b;contact.remove(pair);replaced.append([a,b])
 for g in mapping[old]:
  cp=copy.deepcopy(pair);cp.set('geom1' if a==old else 'geom2',g)
  if cp.get('name'):cp.set('name',cp.get('name')+'_'+g)
  contact.append(cp)
existing={frozenset((x.get('geom1'),x.get('geom2'))) for x in contact.findall('pair')};added=[]
for row in safe:
 a,b=row['geoms']
 for g in mapping[a]:
  if frozenset((g,b)) in existing:raise RuntimeError('Duplicate pair')
  ET.SubElement(contact,'pair',geom1=g,geom2=b,solref='.001 1',solimp='.99 .999 .001');existing.add(frozenset((g,b)));added.append([g,b])
root.set('model','diagnostic_sector_collision_trial');out=p/'diagnostic_sector_collision_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
report={'original_proxy_count':len(mapping),'sector_proxy_count':sum(map(len,mapping.values())),'old_active_pairs_replaced':len(replaced),'new_pairs_for_previously_excluded':len(added),'previously_excluded_source_pairs_restored':len(safe),'compiled_contact_pair_count':int(m.npair),'initial_contacts':int(d.ncon),'replaced_pairs':replaced,
 'limitations':'Sector proxy geometry is conservative but can omit real material; only 56 previously excluded pairs selected. Other exclusions, contact seam duplication, other joints and model uncertainty remain. XML has no runtime range guard. Diagnostic only; not training or hardware.'}
(p/'sector_collision_trial_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print({k:v for k,v in report.items() if k not in ('replaced_pairs','limitations')},flush=True)
