"""Iteratively search simulated open-loop lift trajectories with held pose seeds."""
from pathlib import Path
import argparse,json,time
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

a=argparse.ArgumentParser();a.add_argument('--output-dir',required=True);a.add_argument('--iterations',type=int,default=4);args=a.parse_args()
p=Path(__file__).parent;outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=False)
candidate=json.loads((p/'exploration_clearance_audit.json').read_text())['best'];base=np.asarray(candidate['action_history']);leg=candidate['leg'];nc=8;rep=4;n=nc*rep;rng=np.random.default_rng(8123)
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 mesh=int(env.m.geom_dataid[env.soles[leg]]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts=env.m.mesh_vert[start:start+count].copy();best=None
 for iteration in range(args.iterations):
  candidates=np.clip(base[None,:,:]+rng.normal(0,.12,(nc,len(base),5)),-1,1);candidates[0]=base;schedules=np.repeat(candidates,rep,axis=0)
  # Common random poses make the four-world comparison meaningful.
  env.reset_rng=[np.random.default_rng(8301+i%rep) for i in range(n)];env.reset()
  peak=np.zeros(n);counts=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);last_upright=np.ones(n)
  for step in range(90):
   _,_,done,infos=env.step(schedules[:,min(step//30,len(base)-1),:])
   for i,info in enumerate(infos):
    if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
    elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
    if alive[i]:last_upright[i]=info['upright']
   pos=env.gd.geom_xpos.numpy()[:,env.soles[leg]];rot=env.gd.geom_xmat.numpy()[:,env.soles[leg]].reshape(n,3,3);low=(pos[:,2,None]+rot[:,2,:]@verts.T).min(axis=1)
   support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive
   peak=np.maximum(peak,np.where(support,low,0));counts+=support&(low>.001)
  rows=[]
  for k in range(nc):
   sl=slice(k*rep,(k+1)*rep);safe=alive[sl]&(last_upright[sl]>np.cos(np.deg2rad(10)));ok=safe&(counts[sl]>0)
   score=float(ok.sum()*10+safe.sum()+np.clip(np.mean(peak[sl])*100,0,1))
   rows.append({'candidate':k,'score':score,'successes':int(ok.sum()),'safe':int(safe.sum()),'guards':int(guards[sl].sum()),'falls':int(falls[sl].sum()),'peak_clearance_m':peak[sl].tolist(),'action_history':candidates[k].tolist()})
  winner=max(rows,key=lambda x:x['score']);base=np.asarray(winner['action_history'])
  if best is None or winner['score']>best['score']:best=winner.copy();best['iteration']=iteration
  report={'iteration':iteration,'winner':winner,'best':best,'all_candidates':rows,'scope':'Open-loop heuristic search; four shared initial poses, 1.8 s. No trained gait or hardware safety validation.'}
  (outdir/f'iteration_{iteration:02d}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
  (outdir/'best.json').write_text(json.dumps(report['best'],indent=2),encoding='utf8')
  print('ITERATION '+json.dumps({'iteration':iteration,'successes':winner['successes'],'safe':winner['safe'],'guards':winner['guards'],'score':winner['score']}),flush=True)
finally:env.close()
