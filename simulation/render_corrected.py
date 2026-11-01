import mujoco
from pathlib import Path
from PIL import Image
p=Path(r'C:\Practice_project\biped_sim');m=mujoco.MjModel.from_xml_path(str(p/'biped_with_visual_pins.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);r=mujoco.Renderer(m,480,640);cam=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,cam);cam.lookat[:]=[.04,-.02,.19];cam.distance=.65;cam.azimuth=90;cam.elevation=-4;r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(p/'corrected_pins_preview.png');print('rendered')
