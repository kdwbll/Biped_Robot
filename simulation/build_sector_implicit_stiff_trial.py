"""Build combined implicit-damping and stiff-gear diagnostic model."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent;t=ET.parse(p/'diagnostic_sector_implicit_damping_trial.xml');root=t.getroot();changes=[]
for name in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50'):
 e=root.find(f"./equality/joint[@name='{name}']")
 if e is None or e.get('solref')!='0.001 1':raise RuntimeError('Unexpected gear constraint')
 e.set('solref','0.0005 1');changes.append(name)
root.set('model','diagnostic_sector_implicit_stiff_trial');out=p/'diagnostic_sector_implicit_stiff_trial.xml';t.write(out,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(out))
out={'model':out.name,'modified_equalities':changes,'contact_pairs':int(m.npair),'gear_solref_s':.0005,'physics_timestep_s':float(m.opt.timestep),'limitations':'Unmeasured friction, damping and solver settings. Requires zero external servo hip damping to avoid double counting. Current at 2*timestep reference, not a stability proof. Collision coverage remains incomplete; no XML range guard.'}
(p/'sector_implicit_stiff_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
