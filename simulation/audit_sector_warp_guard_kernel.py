"""Test sticky GPU state guards on synthetic world states; no training."""
from pathlib import Path
import json,sys
import numpy as np,mujoco
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp

from sector_warp_guard import scan_state

wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
h=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)];shaft=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')])
q=np.tile(d.qpos.astype(np.float32),(4,1));v=np.zeros((4,m.nv),dtype=np.float32);q[1,h[0]]+=np.deg2rad(4.6);q[2,h[0]]+=np.deg2rad(.3);v[3,0]=np.nan
with wp.ScopedDevice('cuda:0'):
 gq=wp.array(q,dtype=wp.float32,device='cuda:0');gv=wp.array(v,dtype=wp.float32,device='cuda:0');flags=wp.zeros(4,dtype=wp.int32,device='cuda:0')
 args=[gq,gv,flags,m.nq,m.nv,h[0],h[1],shaft,float(d.qpos[h[0]]),float(d.qpos[h[1]]),float(d.qpos[shaft]),float(np.deg2rad(4.5)),float(np.deg2rad(.2))]
 wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();first=flags.numpy().copy()
 if first.tolist()!=[0,6,4,1]:raise RuntimeError(('unexpected GPU flags',first.tolist()))
 wp.copy(gq,wp.array(np.tile(d.qpos.astype(np.float32),(4,1)),dtype=wp.float32,device='cuda:0'));wp.copy(gv,wp.array(np.zeros((4,m.nv),dtype=np.float32),dtype=wp.float32,device='cuda:0'))
 wp.launch(scan_state,dim=4,inputs=args);wp.synchronize();second=flags.numpy().copy()
 if second.tolist()!=first.tolist():raise RuntimeError('Flags must stay set until reset')
out={'synthetic_flags':first.tolist(),'flags_sticky':bool(np.array_equal(first,second)),'flag_bits':{'1':'nonfinite','2':'hip bound','4':'gear bound'},'limitations':'Synthetic GPU kernel test only. Does not stop/rollback physics, guard contact force, check overflow, synchronize per-step or integrate with training. Thresholds are assumptions, not hardware safety.'}
(p/'sector_warp_guard_kernel_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
