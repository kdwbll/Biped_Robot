"""Isolated compatibility smoke test; does not change ongoing CPU training."""
import os,time,traceback
from pathlib import Path
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

root=Path(__file__).parent
wp.config.kernel_cache_dir=str(root/'warp_cache')
wp.init()
print('Loading original candidate model',flush=True)
model=mujoco.MjModel.from_xml_path(str(root/'biped_filtered_collision_experimental.xml'))
data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
print(f'model nq={model.nq} nv={model.nv} pairs={model.npair} equalities={model.neq}',flush=True)
try:
 with wp.ScopedDevice('cuda:0'):
  start=time.perf_counter();gpu_model=mjw.put_model(model);print('put_model seconds',time.perf_counter()-start,flush=True)
  start=time.perf_counter();gpu_data=mjw.put_data(model,data,nworld=1,nconmax=1024,njmax=4096);print('put_data seconds',time.perf_counter()-start,flush=True)
  start=time.perf_counter();mjw.step(gpu_model,gpu_data);wp.synchronize();print('first_step seconds',time.perf_counter()-start,flush=True)
  q=gpu_data.qpos.numpy();mujoco.mj_step(model,data)
  print('finite',bool(np.isfinite(q).all()),'shape',q.shape,'one_step_max_error',float(np.max(np.abs(q[0]-data.qpos))),flush=True)
  for _ in range(19):mjw.step(gpu_model,gpu_data);mujoco.mj_step(model,data)
  wp.synchronize();q=gpu_data.qpos.numpy()
  print('20_steps max_qpos_error',float(np.max(np.abs(q[0]-data.qpos))),flush=True)
except Exception:
 traceback.print_exc();raise
