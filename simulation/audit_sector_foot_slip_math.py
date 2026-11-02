"""Compare contact-point velocity with MuJoCo Jacobians; no training."""
from pathlib import Path
import json
import mujoco
import numpy as np
from sector_foot_slip import sole_sample_slip

p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);d.qvel[:]=np.random.default_rng(3).normal(0,.05,m.nv);mujoco.mj_forward(m,d)
ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');soles=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{i}_sole_trial_geom') for i in (0,1)]
bodies=[int(m.geom_bodyid[g]) for g in soles];roots=[int(m.body_rootid[b]) for b in bodies];errors=[]
for c in d.contact[:d.ncon]:
    for leg,sole in enumerate(soles):
        if ground not in (c.geom1,c.geom2) or sole not in (c.geom1,c.geom2):continue
        jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,c.pos,bodies[leg]);expected=float(np.linalg.norm((jac@d.qvel)[:2]))
        _,speed=sole_sample_slip([[c.geom1,c.geom2]],[0],[c.pos],[1.],d.cvel[None,...],d.subtree_com[None,...],ground,soles,bodies,roots,1)
        errors.append(abs(float(speed[0,leg])-expected))
if not errors or max(errors)>1e-9:raise RuntimeError(f'Contact Jacobian mismatch: {errors}')
# A point at x=1 rotating about z has velocity +y; a COM velocity of -y cancels it.
cvel=np.zeros((1,1,6));cvel[0,0,2]=1.;cvel[0,0,4]=-1.
_,rolling=sole_sample_slip([[0,1]],[0],[[1.,0.,0.]],[1.],cvel,np.zeros((1,1,3)),0,[1,2],[0,0],[0,0],1)
if rolling[0,0]>1e-12:raise RuntimeError('Pure rotation about contact point should not count as slipping')
out={'contact_points_checked':len(errors),'max_speed_error_mps':max(errors),'pure_rolling_speed_mps':float(rolling[0,0]),
     'limitations':'Instantaneous CPU Jacobian and synthetic rotation checks. Flat stationary ground; sampling and full GPU reward parity require separate tests.'}
(p/'sector_foot_slip_math_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
