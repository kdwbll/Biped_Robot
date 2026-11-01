import mujoco
xml = '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print('MuJoCo', mujoco.__version__, 'step time', d.time)
