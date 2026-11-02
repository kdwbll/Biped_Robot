"""Probe four unactuated MuJoCo-Warp worlds versus CPU; no training."""
from pathlib import Path
import json,sys,time,traceback
import numpy as np,mujoco
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
out={'model':'diagnostic_sector_implicit_stiff_trial.xml','nworld':4,'training_started':False,'samples':[]}
try:
 import warp as wp
 import mujoco_warp as mjw
 wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/out['model']));m.opt.timestep=.00025
 c=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,c,0);mujoco.mj_forward(m,c)
 with wp.ScopedDevice('cuda:0'):
  gm=mjw.put_model(m);gd=mjw.put_data(m,c,nworld=4,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize()
  with wp.ScopedCapture() as capture:mjw.step(gm,gd)
  graph=capture.graph;start=time.perf_counter()
  for step in range(1,401):
   wp.capture_launch(graph);mujoco.mj_step(m,c)
   if step in (1,20,80,200,400):
    wp.synchronize();q=gd.qpos.numpy();v=gd.qvel.numpy();overflow=gd.overflow.numpy()
    if not np.isfinite(q).all() or not np.isfinite(v).all() or np.any(overflow):raise RuntimeError('GPU nonfinite or capacity overflow')
    if not np.isfinite(c.qpos).all() or not np.isfinite(c.qvel).all():raise RuntimeError('CPU nonfinite')
    row={'step':step,'time_s':float(c.time),'max_qpos_error':float(np.max(np.abs(q-c.qpos[None,:]))),'max_qvel_error':float(np.max(np.abs(v-c.qvel[None,:]))),'world_qpos_spread':float(np.max(np.ptp(q,axis=0))),'overflow':overflow.tolist()};out['samples'].append(row);print(row,flush=True)
  out['wall_seconds_including_cpu_and_sync']=time.perf_counter()-start
 out['success']=True
except Exception as e:
 out.update({'success':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
out['limitations']='Four identical unactuated worlds for only 0.1 s; no servo, force guard, policy, PPO or training. One initialization and limited duration cannot establish parity or throughput. CPU float64 and GPU float32 can diverge.'
(p/'sector_warp_unactuated_parity.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('success',out['success'],flush=True)
