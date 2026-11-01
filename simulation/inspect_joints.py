import mujoco
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_chassis_collision_candidate.xml');print('nq',m.nq,'nu',m.nu)
for i in range(m.njnt):print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i),m.jnt_qposadr[i])
