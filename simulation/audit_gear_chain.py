"""Check candidate gearing signs with a tiny in-memory coordinate perturbation."""
from pathlib import Path
import json
import mujoco
import numpy as np

p = Path(__file__).parent
m = mujoco.MjModel.from_xml_path(str(p/'biped_fresh_mass_only_candidate.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0)
mujoco.mj_forward(m,d)
names = ['hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate']
ids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
axes = [d.xaxis[i].copy() for i in ids]
anchors = [d.xanchor[i].copy() for i in ids]
eqnames = ['opposite_world_rotation_assumption','estimated_pinion_23_gear_50']
eqids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,n) for n in eqnames]
def residuals():
    return [float(d.efc_pos[np.flatnonzero((d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (d.efc_id == i))[0]]) for i in eqids]
before = residuals()
delta = np.array([.001,-.00046,-.00046])
for i,x in zip(ids,delta):
    d.qpos[m.jnt_qposadr[i]] += x
mujoco.mj_forward(m,d)
after = residuals()
axis_signs = [float(np.dot(axis,axes[0])) for axis in axes]
signed_world = [float(x*s) for x,s in zip(delta,axis_signs)]
distances=[]
for anchor in anchors[1:]:
    offset=anchor-anchors[0]
    distances.append(float(np.linalg.norm(offset-np.dot(offset,axes[0])*axes[0])))
out = {'model':'biped_fresh_mass_only_candidate.xml', 'perturbation_rad':dict(zip(names,delta.tolist())),
       'axis_dot_with_pinion':dict(zip(names,axis_signs)),
       'world_axis_signed_angular_delta_rad':dict(zip(names,signed_world)),
       'candidate_gear_equality_residual_change_rad':dict(zip(eqnames,(np.array(after)-before).tolist())),
       'pinion_to_candidate_gear_axis_distances_m':dict(zip(names[1:],distances)),
       'limitations':'L0 anatomical left/right mapping and tooth counts remain to be verified. '
                     'This is a local constraint check, not a driven motion or physical validation.'}
(p/'gear_chain_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('signed common-axis deltas:',dict(zip(names,np.round(signed_world,8))))
print('constraint residual changes:',np.round(np.array(after)-before,12))
print('pinion to gear axis distances mm:',np.round(np.array(distances)*1000,3))
