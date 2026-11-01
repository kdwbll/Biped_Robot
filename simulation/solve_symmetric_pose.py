from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
p=Path(r'C:\Practice_project\biped_sim');m=mujoco.MjModel.from_xml_path(str(p/'biped_chassis_collision_candidate.xml'));d=mujoco.MjData(m);q0=d.qpos.copy(); gids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,x) for x in ['L0_foot_proxy','L1_foot_proxy']];robot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');idx=np.array([i for i in range(7,m.nq) if i not in (7,26,45)])
rotinds=[m.jnt_qposadr[i] for i in range(m.njnt) if mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) in ['L0_part6_r','L1_part6_r']]
def evaluate(x,verbose=False):
 d.qpos[:]=q0;d.qpos[idx]=x;mujoco.mj_forward(m,d)
 eq=np.array([d.efc_pos[i] for i in range(d.nefc) if d.efc_type[i]==mujoco.mjtConstraint.mjCNSTR_EQUALITY]);a,b=d.geom_xpos[gids];com=d.subtree_com[robot];goal=np.array([(a[0]+b[0])/2-com[0],a[0]-b[0],(a[1]+b[1])/2-com[1],a[2]-b[2],d.qpos[rotinds[0]],d.qpos[rotinds[1]]]);
 if verbose:return eq,goal,a.copy(),b.copy(),com.copy()
 return np.r_[eq*1000,goal[[0,1,3,4,5]]*1000,x*.1]
res=least_squares(evaluate,q0[idx],max_nfev=300,xtol=1e-10,ftol=1e-10,gtol=1e-10); eq,goal,a,b,com=evaluate(res.x,True);print('success',res.success,'nfev',res.nfev,'maximum equality',max(abs(eq)),'goal errors',goal);print('left right',a,b,'center of mass',com);print('maximum joint shift',max(abs(res.x)))
np.savez(p/'balanced_initial_pose.npz',qpos=d.qpos.copy(),eq=eq,goal=goal,foot0=a,foot1=b,com=com)


