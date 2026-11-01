import mujoco
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_shaft_collision_trial.xml');d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
for i in range(m.nu):print(i,mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i),'length',d.actuator_length[i], 'vel',d.actuator_velocity[i], 'ctrlrange',m.actuator_ctrlrange[i])
