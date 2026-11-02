"""One-world MuJoCo Warp conversion and one-step parity probe; never trains."""
from pathlib import Path
import json,sys,time,traceback
import numpy as np
import mujoco
p=Path(__file__).parent
local=p/'warp_test_packages'
if local.exists():sys.path.insert(0,str(local))
report={'model':'diagnostic_sector_implicit_stiff_trial.xml','nworld':1,'training_started':False}
try:
 import warp as wp
 import mujoco_warp as mjw
 wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init()
 m=mujoco.MjModel.from_xml_path(str(p/report['model']));cpu=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,cpu,0);mujoco.mj_forward(m,cpu)
 with wp.ScopedDevice('cuda:0'):
  t=time.perf_counter();gm=mjw.put_model(m);gd=mjw.put_data(m,cpu,nworld=1,nconmax=4096,njmax=8192);mjw.forward(gm,gd);wp.synchronize();report['conversion_forward_s']=time.perf_counter()-t
  before=gd.qpos.numpy()[0].copy();t=time.perf_counter();mjw.step(gm,gd);wp.synchronize();report['one_step_wall_s']=time.perf_counter()-t
  after=gd.qpos.numpy()[0].copy();vel=gd.qvel.numpy()[0].copy();overflow=gd.overflow.numpy().copy()
 mujoco.mj_step(m,cpu);report.update({'success':True,'qpos_max_difference_one_step':float(np.max(np.abs(cpu.qpos-after))),'qvel_max_difference_one_step':float(np.max(np.abs(cpu.qvel-vel))),'gpu_step_max_qpos_change':float(np.max(np.abs(after-before))),'gpu_finite':bool(np.isfinite(after).all() and np.isfinite(vel).all()),'overflow':overflow.tolist(),'gpu_name':str(wp.get_device('cuda:0'))})
except Exception as e:
 report.update({'success':False,'error':str(e),'error_type':type(e).__name__,'traceback':traceback.format_exc()})
report['limitations']='One GPU world and one unactuated physics step; no controller, force guard, batched rollout, PPO or training. One-step agreement alone is not useful parity validation. Memory and runtime scale with worlds/contacts.'
(p/'sector_warp_conversion_probe.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
