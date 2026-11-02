"""Check actual sole mesh minimum height for one open-loop action in GPU simulation."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;env=SectorWarpStandingVecEnv(nworld=1,episode_seconds=4.,on_guard='terminate',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
    env.reset();verts=[]
    for geom in env.soles:
        mesh=int(env.m.geom_dataid[geom]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[start:start+count].copy())
    max_clearance=np.full(2,-1.);min_clearance=np.full(2,1.);max_contact=np.zeros(2);unloaded=np.zeros(2,int);guard=0
    for step in range(180):
        action=np.zeros((1,5)) if step<25 else np.array([[1,1,0,1,0.]])
        _,_,done,infos=env.step(action);info=infos[0]
        if done[0] and info['guard_flags']:guard+=1;break
        positions=env.gd.geom_xpos.numpy()[0,env.soles];rotations=env.gd.geom_xmat.numpy()[0,env.soles].reshape(2,3,3)
        for leg in (0,1):
            world_z=positions[leg,2]+verts[leg]@rotations[leg,2,:]
            clearance=float(np.min(world_z));max_clearance[leg]=max(max_clearance[leg],clearance);min_clearance[leg]=min(min_clearance[leg],clearance)
        forces=np.asarray(info['sampled_sole_peak_normal_n']);max_contact=np.maximum(max_contact,forces)
        if step>=50:unloaded+=(forces<.5)
    out={'action':[1,1,0,1,0],'max_sole_mesh_min_z_m':max_clearance.tolist(),'min_sole_mesh_min_z_m':min_clearance.tolist(),
         'sampled_unloaded_frames':unloaded.tolist(),'max_single_contact_force_n':max_contact.tolist(),'guard_events':guard,
         'limitations':'One open-loop action, mesh minimum relative to assumed ground z=0, 20 ms sampling; not gait or hardware validation.'}
    (p/'foot_clearance_probe_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
