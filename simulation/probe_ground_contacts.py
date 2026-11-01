"""Identify non-foot ground contacts during the assisted reference motion."""
from collections import Counter
import json
import mujoco
from supported_stepping_env import SupportedSteppingEnv
from pretrain_harness_teacher import action_for

e=SupportedSteppingEnv(episode_seconds=4.,disturbance=0.,support_gain=3.,weight_support=.8)
e.reset(seed=0);counts=Counter();min_dist={}
for k in range(200):
    _,_,term,trunc,_=e.step(action_for(k))
    for c in e.data.contact[:e.data.ncon]:
        names=[mujoco.mj_id2name(e.model,mujoco.mjtObj.mjOBJ_GEOM,n) or '' for n in (c.geom1,c.geom2)]
        if 'ground' in names:
            other=names[1] if names[0]=='ground' else names[0]
            counts[other]+=1;min_dist[other]=min(min_dist.get(other,0.),float(c.dist))
    if term or trunc:break
e.close()
out=dict(contact_points=dict(counts),minimum_distance_m=min_dist)
with open('assisted_ground_contacts.json','w',encoding='utf8') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
