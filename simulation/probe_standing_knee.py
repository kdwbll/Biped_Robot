"""Measure selected linkage pivot angles in the candidate initial pose only."""
import json
import mujoco
import numpy as np
from biped_env import BipedEnv

e=BipedEnv(episode_seconds=4.);e.reset(seed=0)
def anchor(name):
    eq=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_EQUALITY,name)
    body=e.model.eq_obj1id[eq];local=e.model.eq_data[eq,:3]
    return e.data.xpos[body]+e.data.xmat[body].reshape(3,3)@local
out={}
for leg,names in ((0,('L0_pin0','L0_pin5','L0_pin7')),(1,('L1_pin1','L1_pin6','L1_pin5'))):
    p=[anchor(name) for name in names];u=p[0]-p[1];v=p[2]-p[1]
    angle=np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
    out[str(leg)]=dict(pivots=names,world_points_m=[x.tolist() for x in p],included_angle_deg=float(angle),bend_from_straight_deg=float(180-angle))
e.close();print(json.dumps(out,indent=2))
