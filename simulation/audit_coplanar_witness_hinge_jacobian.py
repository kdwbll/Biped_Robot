"""Check candidate coplanar witness normal motion under ideal hinge kinematics."""
from pathlib import Path
import json
import mujoco
import numpy as np
p=Path(__file__).parent
source=json.loads((p/'remaining_collision_cad_witness_audit.json').read_text(encoding='utf8'))['results']
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_restored_near_pairs_trial.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');R=d.xmat[root].reshape(3,3);q0=d.qpos.copy();joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)];shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft');rows=[]
for r in source:
 a=r['chassis_cad_face'];b=r['tower_cad_face'];na=np.array(a['normal']);nb=np.array(b['normal'])
 if not(a['planar'] and b['planar'] and max(a['gap_mm'],b['gap_mm'])<.01 and np.dot(na,nb)<-.999):continue
 ga,gb=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in r['geoms']]
 for angle in (-5.,0.,5.):
  d.qpos[:]=q0;d.qpos[m.jnt_qposadr[shaft]]+=np.deg2rad(-2*angle)
  for joint in joints:d.qpos[m.jnt_qposadr[joint]]+=np.deg2rad(angle)
  mujoco.mj_forward(m,d);point=d.xanchor[joints[r['leg']]].copy();normal=R@na
  jac1=np.zeros((3,m.nv));jac2=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac1,None,point,m.geom_bodyid[ga]);mujoco.mj_jac(m,d,jac2,None,point,m.geom_bodyid[gb]);j=normal@(jac2-jac1)
  rows.append({'leg':r['leg'],'geoms':r['geoms'],'hip_angle_deg':angle,'relative_normal_jacobian_max_abs':float(np.max(np.abs(j))),'relative_normal_jacobian_l2':float(np.linalg.norm(j))})
if not rows:raise RuntimeError('No selected coplanar examples')
out={'sample_count':len(rows),'results':rows,'limitations':'Local velocity Jacobian at three sampled hip angles for selected initially near-coplanar witnesses. Not force, bearing preload, real contact friction, or finite-motion proof; mapping and mesh approximations remain.'}
(p/'coplanar_witness_hinge_jacobian_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('sampled',len(rows),'max abs J',max(x['relative_normal_jacobian_max_abs'] for x in rows),flush=True)
