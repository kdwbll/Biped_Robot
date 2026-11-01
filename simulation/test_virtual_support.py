"""Probe lift with training-only balance assistance, not a policy benchmark."""
import itertools
import json
import numpy as np
from supported_stepping_env import SupportedSteppingEnv

rows = []
for gain in (0.0, 3.0):
    for leg, (a, b) in itertools.product((0, 1), ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))):
        env = SupportedSteppingEnv(episode_seconds=2., disturbance=0., support_gain=gain)
        env.reset(seed=0)
        action = np.zeros(5)
        action[2*leg:2*leg+2] = [a, b]
        peak = 0.
        for k in range(100):
            _, _, term, trunc, info = env.step(action)
            peak = max(peak, info['foot_clearance_m'][leg])
            if term or trunc:
                break
        rows.append(dict(gain=gain, leg=leg, action=[a,b], peak_mm=peak*1000,
                         seconds=(k+1)*.02, fell=bool(term),
                         tilt_deg=float(np.degrees(np.arccos(np.clip(info['upright'],-1,1))))))
        env.close()
        print(json.dumps(rows[-1]), flush=True)
with open('virtual_support_probe.json', 'w', encoding='utf8') as f:
    json.dump(rows, f, indent=2)
