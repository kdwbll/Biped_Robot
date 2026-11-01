"""Free-space kinematic diagnostic, not a realistic walking evaluation."""
import itertools, json
import numpy as np
import mujoco
from stepping_env import SteppingEnv

rows=[]
for leg, (a,b) in itertools.product((0,1), ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))):
    e=SteppingEnv(episode_seconds=1.5,disturbance=0.); e.reset(seed=0)
    e.model.opt.gravity[:]=0; e.data.qpos[2]+=.20; mujoco.mj_forward(e.model,e.data)
    start=e.data.geom_xpos[e.foot_geom[leg],2]-e.data.xipos[e.body,2]
    action=np.zeros(5); action[2*leg:2*leg+2]=[a,b]
    amin=e.data.actuator_length.copy(); max_lift=0.; max_move=0.
    for _ in range(75):
        _,_,term,trunc,_=e.step(action)
        diff=e.data.geom_xpos[e.foot_geom[leg],2]-e.data.xipos[e.body,2]-start
        max_lift=max(max_lift,float(diff)); max_move=max(max_move,float(np.max(np.abs(e.data.actuator_length-amin))))
        if term or trunc: break
    row=dict(leg=leg,action=[a,b],max_relative_up_mm=max_lift*1000,actuator_displacement_rad=max_move,terminated=bool(term))
    rows.append(row); e.close(); print(json.dumps(row),flush=True)
with open('leg_kinematic_probe.json','w',encoding='utf8') as f: json.dump(rows,f,indent=2)
