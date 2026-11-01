import mujoco
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_with_visual_pins.xml')
for i in range(m.ngeom):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)
 if name and ('visual' in name or 'proxy' in name or 'ground' in name):print(name,'type',m.geom_type[i],'contact',m.geom_contype[i],m.geom_conaffinity[i])
