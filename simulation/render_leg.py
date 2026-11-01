from pathlib import Path
import mujoco
from PIL import Image
p=Path(__file__).parent; m=mujoco.MjModel.from_xml_path(str(p/'single_leg_mesh_experiment.xml')); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,480,640); r.update_scene(d,camera='side'); Image.fromarray(r.render()).save(p/'single_leg_preview.png'); print('saved')

