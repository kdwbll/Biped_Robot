from pathlib import Path
import mujoco
import mujoco.viewer
p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'biped_centered_initial_pose.xml'))
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0)
mujoco.mj_forward(m,d)
mujoco.viewer.launch(m,d)
