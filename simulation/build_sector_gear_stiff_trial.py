"""Create a gear-stiffness diagnostic without changing the source model."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent;tree=ET.parse(p/'diagnostic_sector_collision_trial.xml');root=tree.getroot();changes=[]
for name in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50'):
 eq=root.find(f"./equality/joint[@name='{name}']")
 if eq is None or eq.get('solref')!='0.001 1':raise RuntimeError('Unexpected gear constraint')
 changes.append({'constraint':name,'old':eq.get('solref'),'new':'0.0005 1'});eq.set('solref','0.0005 1')
root.set('model','diagnostic_sector_gear_stiff_trial');out=p/'diagnostic_sector_gear_stiff_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));report={'changes':changes,'timestep_s':float(m.opt.timestep),'minimum_refsafety_timestep_multiple':2,'limitations':'Hypothetical solver stiffness, not measured gear mechanics; at the 2*timestep reference setting; this does not prove stability. Short tests required; only this XML carries change. Not for training or hardware.'}
(p/'sector_gear_stiff_trial_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
