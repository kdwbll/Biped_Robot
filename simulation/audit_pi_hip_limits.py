"""Check simulated hip limit violations during the neutral PI-D experiment."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
gear_candidate='--gear' in sys.argv
m=mujoco.MjModel.from_xml_path(str(p/('biped_standing_gear_candidate.xml' if gear_candidate else 'biped_fresh_mass_only_candidate.xml')));m.opt.timestep=.00025
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
names=['L0_hip_candidate','L1_hip_candidate'];ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
mins=np.full(2,np.inf);maxs=np.full(2,-np.inf);max_excess=np.zeros(2)
for _ in range(round(6/m.opt.timestep)):
 s.step()
 for k,j in enumerate(ids):
  q=d.qpos[m.jnt_qposadr[j]];lo,hi=m.jnt_range[j]
  mins[k]=min(mins[k],q);maxs[k]=max(maxs[k],q);max_excess[k]=max(max_excess[k],lo-q,q-hi)
out={'duration_s':6,'notes':'Soft limits in candidate are unverified and may not represent real mechanism.',
     'joints':[{'name':name,'range_deg':np.rad2deg(m.jnt_range[j]).tolist(),
                'observed_deg':np.rad2deg([mins[k],maxs[k]]).tolist(),
                'max_outside_range_deg':float(np.rad2deg(max_excess[k]))}
               for k,(name,j) in enumerate(zip(names,ids))]}
(p/('standing_gear_pi_hip_limit_audit.json' if gear_candidate else 'standing_pi_hip_limit_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
for x in out['joints']:print(x['name'],'limit',np.round(x['range_deg'],3),'observed',np.round(x['observed_deg'],3),'max exceed deg',round(x['max_outside_range_deg'],3))
