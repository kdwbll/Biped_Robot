"""Check candidate actuator independence in the current keyframe, without training."""
from pathlib import Path
import json
import mujoco
import numpy as np

p = Path(__file__).parent
m = mujoco.MjModel.from_xml_path(str(p / 'biped_filtered_collision_experimental.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
# Equality-only local tangent space: deliberately exclude floor contacts and soft limits.
jac = np.asarray(d.efc_J).reshape(d.nefc, m.nv)
eq = jac[d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]
_, singular, vt = np.linalg.svd(eq, full_matrices=True)
rank = int(np.linalg.matrix_rank(eq, tol=1e-6))
null = vt[rank:].T
moment = np.zeros((m.nu, m.nv))
for i in range(m.nu):
    start, count = int(d.moment_rowadr[i]), int(d.moment_rownnz[i])
    cols = d.moment_colind[start:start+count]
    moment[i, cols] = d.actuator_moment[start:start+count]
projected = moment @ null
out = {'model': 'biped_filtered_collision_experimental.xml', 'state': 'keyframe 0',
       'scope': 'Local linearized equality kinematics; ignores contacts, limits and physical actuator mounting. '
                'Rank depends on tolerance and mixed units; it is a numerical diagnostic, not proof of five physical DOFs.',
       'nv': m.nv, 'equality_rows': eq.shape[0], 'equality_rank': rank, 'tangent_dimension': null.shape[1],
       'rank_tolerance': 1e-6,
       'rank_by_tolerance': {str(tol): int(np.linalg.matrix_rank(eq, tol=tol)) for tol in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)},
       'actuators': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)],
       'projected_actuator_rank': int(np.linalg.matrix_rank(projected, tol=1e-8)),
       'projected_actuator_singular_values': np.linalg.svd(projected, compute_uv=False).tolist(),
       'equality_singular_values': singular.tolist()}
(p / 'drive_kinematics_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('equality rank', rank, 'tangent dimension', null.shape[1], 'actuator rank', out['projected_actuator_rank'])
print('actuators', out['actuators'])
print('projected singular values', np.round(out['projected_actuator_singular_values'], 6))
