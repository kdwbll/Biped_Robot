"""GPU batch microbenchmark only: no servo controller or PPO, not a training speed."""
import sys,time
from pathlib import Path
import mujoco,numpy as np,warp as wp,mujoco_warp as mjw
root=Path(__file__).parent;wp.config.kernel_cache_dir=str(root/'warp_cache');wp.init()
n=int(sys.argv[1]);steps=int(sys.argv[2]) if len(sys.argv)>2 else 100;m=mujoco.MjModel.from_xml_path(str(root/'biped_filtered_collision_experimental.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=n,nconmax=1024,njmax=4096)
 print('allocated',n,'worlds; compiling/capturing',flush=True)
 mjw.step(gm,gd);wp.synchronize()
 with wp.ScopedCapture() as capture:mjw.step(gm,gd)
 graph=capture.graph
 for _ in range(5):wp.capture_launch(graph)
 wp.synchronize();start=time.perf_counter()
 for _ in range(steps):wp.capture_launch(graph)
 wp.synchronize();elapsed=time.perf_counter()-start
 print('RESULT',dict(worlds=n,physics_steps=steps*n,seconds=elapsed,physics_steps_per_sec=steps*n/elapsed,finite=bool(np.isfinite(gd.qpos.numpy()).all())),flush=True)
