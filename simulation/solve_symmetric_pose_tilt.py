from pathlib import Path
import math,mujoco,numpy as np
from scipy.optimize import least_squares
p=Path(r'C:\Practice_project\biped_sim');m=mujoco.MjModel.from_xml_path(str(p/'biped_chassis_collision_candidate.xml'));d=mujoco.MjData(m);qbase=np.load(p/'balanced_initial_pose.npz')['qpos'];idx=np.array([i for i in range(7,m.nq) if i not in (7,26,45)]);foot=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('L0_visual6','L1_visual6')];proxy=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('L0_foot_proxy','L1_foot_proxy')];robot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
def fn(x,ret=False):
 d.qpos[:]=qbase; d.qpos[idx]=x[:-1];r=x[-1];d.qpos[3:7]=[math.cos(r/2),math.sin(r/2),0,0];mujoco.mj_forward(m,d);eq=np.array([d.efc_pos[i] for i in range(d.nefc) if d.efc_type[i]==mujoco.mjtConstraint.mjCNSTR_EQUALITY]);a,b=d.geom_xpos[proxy];com=d.subtree_com[robot]; lows=[]
 for g in foot:
  mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];lows.append(v[:,2].min())
 goal=np.array([(a[0]+b[0])/2-com[0],a[0]-b[0],(a[1]+b[1])/2-com[1],lows[0]-lows[1]])
 if ret:return eq,goal,com.copy(),a.copy(),b.copy(),lows,d.qpos.copy()
 return np.r_[eq*1000,goal*1000,(x[:-1]-qbase[idx])*.1,r*.1]
x=np.r_[qbase[idx],0.0];res=least_squares(fn,x,max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10);eq,goal,com,a,b,lows,q=fn(res.x,True);q[2]-=min(lows);print('success',res.success,'roll degrees',res.x[-1]*180/math.pi,'goal mm',goal*1000,'closure mm',max(abs(eq))*1000,'foot low',lows,'COM',com);np.savez(p/'balanced_initial_pose_exact.npz',qpos=q,goal=goal,roll=res.x[-1])
