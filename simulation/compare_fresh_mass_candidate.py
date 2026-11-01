"""Compare neutral standing under assumptions, without training or support."""
from pathlib import Path
import json
import mujoco
import numpy as np
from stabilized_servo import StabilizedServo

p = Path(__file__).parent
out = []
for filename in ('biped_filtered_collision_experimental.xml', 'biped_fresh_mass_only_candidate.xml'):
    m = mujoco.MjModel.from_xml_path(str(p/filename))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot_free')
    controller = StabilizedServo(m,d)
    z0 = float(d.xipos[base,2])
    com0 = np.average(d.xipos[1:], axis=0, weights=m.body_mass[1:])
    max_speed = max_torque = max_tilt = 0.0
    for i in range(round(2.0/m.opt.timestep)):
        controller.step()
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise RuntimeError(f'non-finite state for {filename} at {d.time}')
        max_speed = max(max_speed, float(np.max(np.abs(d.qvel))))
        max_torque = max(max_torque, float(np.max(np.abs(d.ctrl))))
        z_axis = d.xmat[base].reshape(3,3)[2,2]
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(z_axis,-1,1)))))
    row = {'model': filename, 'total_mass_kg': float(mujoco.mj_getTotalmass(m)),
           'initial_com_m': com0.tolist(), 'initial_base_com_height_m': z0,
           'final_base_com_height_m': float(d.xipos[base,2]), 'height_loss_mm': float((z0-d.xipos[base,2])*1000),
           'max_sampled_each_substep_qvel_mixed_units': max_speed, 'max_sampled_each_substep_torque_nm': max_torque,
           'max_tilt_deg': max_tilt, 'seconds': float(d.time)}
    out.append(row)
    print(filename, 'height loss mm', round(row['height_loss_mm'],3),
          'max tilt deg',round(max_tilt,3),'max speed mixed units',round(max_speed,3))
(p/'fresh_mass_neutral_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
