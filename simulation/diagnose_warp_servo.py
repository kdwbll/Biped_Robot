"""Print early CPU/GPU servo signals for a neutral command."""
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from theoretical_servo import TheoreticalServoController
from warp_servo import WarpServo
wp.config.kernel_cache_dir=str(Path(__file__).parent/'warp_cache');wp.init()
m=mujoco.MjModel.from_xml_path('biped_filtered_collision_experimental.xml');d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);cpu=TheoreticalServoController(m,d)
with wp.ScopedDevice('cuda:0'):
 gm=mjw.put_model(m);gd=mjw.put_data(m,d,nworld=1,nconmax=1024,njmax=4096)
 servo=WarpServo(gm,gd,cpu.initial_positions,1);servo.set_actions(np.zeros((1,5)))
 for k in range(40):
  wp.capture_launch(servo.graph);cpu.step()
  if k in (0,1,2,4,9,19,39):
   wp.synchronize()
   print('step',k+1,'cpu_maxv',np.max(np.abs(d.qvel)),'gpu_maxv',np.max(np.abs(gd.qvel.numpy())),
    'cpu_ctrl',d.ctrl,'gpu_ctrl',gd.ctrl.numpy()[0],
    'cpu_len',d.actuator_length,'gpu_len',gd.actuator_length.numpy()[0],flush=True)
