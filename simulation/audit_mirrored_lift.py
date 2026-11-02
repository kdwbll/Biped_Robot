"""Screen possible leg-mirroring signs against one common simulated pose."""
from pathlib import Path
import itertools,json,sys
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;source=json.loads((p/'训练记录'/'lift_search_20261102_a'/'best.json').read_text());history=np.asarray(source['action_history'])
central_modes=[-1.] if len(sys.argv)>1 and sys.argv[1]=='minus1' else [0.,1.]
signs=np.tile(np.asarray(list(itertools.product((-1.,1.),repeat=4))),(len(central_modes),1));central=np.repeat(central_modes,16);n=len(signs)
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset_rng=[np.random.default_rng(7317) for _ in range(n)];env.reset()
 mesh=int(env.m.geom_dataid[env.soles[0]]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts=env.m.mesh_vert[start:start+count].copy()
 peak=np.zeros(n);frames=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);upright=np.ones(n);pairs=[None]*n
 for step in range(90):
  ref=history[min(step//30,len(history)-1)];action=np.empty((n,5));action[:,:4]=ref[[2,3,0,1]][None,:]*signs;action[:,4]=central*ref[4]
  _,_,done,infos=env.step(action)
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags'] and alive[i]:guards[i]+=1;alive[i]=False;pairs[i]=info['sampled_peak_self_contact_pair']
   elif done[i] and not info['TimeLimit.truncated'] and alive[i]:falls[i]+=1;alive[i]=False
   if alive[i]:upright[i]=info['upright']
  pos=env.gd.geom_xpos.numpy()[:,env.soles[0]];rot=env.gd.geom_xmat.numpy()[:,env.soles[0]].reshape(n,3,3);low=(pos[:,2,None]+rot[:,2,:]@verts.T).min(axis=1)
  supported=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1]>.5 for x in infos])&alive
  peak=np.maximum(peak,np.where(supported,low,0));frames+=supported&(low>.001)
 rows=[]
 for i in range(n):
  rows.append({'signs':signs[i].tolist(),'central_sign':float(central[i]),'supported_peak_clearance_m':float(peak[i]),'supported_frames_over_1mm':int(frames[i]),
               'guard_count':int(guards[i]),'fall_count':int(falls[i]),'guard_pair':pairs[i],'final_upright':float(upright[i]),
               'success':bool(alive[i] and frames[i]>0 and upright[i]>np.cos(np.deg2rad(10)))})
 out={'seed':7317,'mirrored_leg':0,'results':rows,'limitations':'One common initial pose, speculative sign mappings, one open-loop sequence, 1.8 s. Symmetry is unverified in hardware; not training or walking.'}
 (p/('mirrored_lift_repeat_minus1_audit.json' if central_modes==[-1.] else 'mirrored_lift_additional_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'successes':[x for x in rows if x['success']],'guard_count':int(guards.sum()),'fall_count':int(falls.sum())},indent=2),flush=True)
finally:env.close()
