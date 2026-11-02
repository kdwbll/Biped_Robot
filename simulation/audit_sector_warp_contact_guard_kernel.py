"""Test GPU contact guard classification on synthetic contacts; no training."""
from pathlib import Path
import json,sys,numpy as np,mujoco
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
from sector_warp_guard import scan_contacts
wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
other=[i for i in range(m.ngeom) if i!=ground][:2];geom=np.array([[ground,other[0]],[other[0],other[1]],[other[0],other[1]],[other[0],other[1]],[other[0],other[1]]],dtype=np.int32);world=np.array([0,1,2,3,2],dtype=np.int32);force=np.zeros((5,6),dtype=np.float32);force[:,0]=[100,21,19,25,20]
with wp.ScopedDevice('cuda:0'):
 g=wp.array(geom,dtype=wp.vec2i,device='cuda:0');w=wp.array(world,dtype=wp.int32,device='cuda:0');f=wp.array(force,dtype=wp.spatial_vector,device='cuda:0');flags=wp.array(np.array([0,0,0,1],dtype=np.int32),dtype=wp.int32,device='cuda:0')
 wp.launch(scan_contacts,dim=5,inputs=[g,w,f,flags,5,4,ground,20.]);wp.synchronize();result=flags.numpy()
if result.tolist()!=[0,8,0,9]:raise RuntimeError(('unexpected flags',result.tolist()))
out={'synthetic_flags':result.tolist(),'floor_excluded':True,'at_threshold_not_triggered':True,'state_flag_preserved':True,'limit_n':20.,'limitations':'Synthetic forces, one-time kernel. Does not validate live GPU contact-force extraction, per-world resets, training, physical limits, or sampling completeness.'}
(p/'sector_warp_contact_guard_kernel_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
