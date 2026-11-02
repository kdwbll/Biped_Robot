"""Check sampled ground contact and foot displacement diagnostics without learning."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
gpu=SectorWarpStandingVecEnv(nworld=4,episode_seconds=2.)
cpu=SectorStandingEnv(episode_seconds=2.,model_variant='implicit_stiff',max_sampled_self_contact_n=20.,
                      max_gear_residual_deg=.2,max_pwm_delta_us_per_control=1.)
try:
    gpu.reset();cpu.reset(seed=1);gpu_peak=np.zeros((4,2));cpu_peak=np.zeros(2);max_delta=0.
    for _ in range(50):
        _,_,gd,gi=gpu.step(np.zeros((4,5)))
        _,_,ct,cu,ci=cpu.step(np.zeros(5))
        if gd.any() or ct or cu:raise RuntimeError('Unexpected termination')
        gpu_peak=np.maximum(gpu_peak,[x['sampled_sole_peak_normal_n'] for x in gi])
        cpu_peak=np.maximum(cpu_peak,ci['sampled_sole_peak_normal_n'])
        delta=np.asarray(gi[0]['foot_body_delta_xy_m'])
        if delta.shape!=(2,2) or not np.isfinite(delta).all():raise RuntimeError('Invalid foot displacement')
        max_delta=max(max_delta,float(np.max(np.abs(delta))))
    if np.any(gpu_peak<=0) or np.any(cpu_peak<=0):raise RuntimeError('Expected both feet to make sampled ground contact')
    out={'frames':50,'gpu_worlds':4,'gpu_peak_per_contact_n':gpu_peak.tolist(),
         'cpu_peak_per_contact_n':cpu_peak.tolist(),'max_gpu_foot_body_delta_per_frame_m':max_delta,
         'limitations':'Neutral standing only; peak is per single sampled contact, not total foot load. Contact sampled every 2.5 ms; body displacement is not sole slip distance during contact. No training or hardware validation.'}
    (p/'sector_foot_telemetry_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:cpu.close();gpu.close()
