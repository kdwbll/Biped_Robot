"""Inspect neutral controller posture and support across time without training."""
import json
import numpy as np
from stabilized_servo import StabilizedServo
from supported_stepping_env import SupportedSteppingEnv

for weight in (0.,.35):
 e=SupportedSteppingEnv(episode_seconds=4.,disturbance=0.,support_gain=3.75*weight,weight_support=weight)
 e.reset(seed=1);e.controller=StabilizedServo(e.model,e.data)
 for k in range(201):
  if k in (0,1,5,25,50,100,200):
   print(json.dumps(dict(weight=weight,seconds=k*.02,body_height_m=float(e.data.xipos[e.body,2]),
    actuator_angles_rad=(e.data.actuator_length-e.controller.initial_positions).tolist(),
    motor_target_rad=(e.controller.target_positions-e.controller.initial_positions).tolist())),flush=True)
  if k<200:e.step(np.zeros(5))
 e.close()
