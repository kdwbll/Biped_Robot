"""Summarize training-readiness evidence; does not start training."""
from pathlib import Path
import json
p=Path(__file__).parent
def read(name):return json.loads((p/name).read_text(encoding='utf8'))
smoke=read('sector_standing_env_smoke_audit.json');axes=read('sector_standing_env_single_axis_audit.json')['results'];peak=read('sector_peak_contact_direction_audit.json')['peak'];near=read('chassis0_finer_proxy_range_audit.json')['unresolved_within_0_1mm_count'];first=read('restored_near_pairs_build.json')['restored_pair_count'];second=read('sector_collision_trial_build.json')['previously_excluded_source_pairs_restored']
checks=('neutral_1s_completed','neutral_reproducible','contact_and_gear_metrics_finite','invalid_action_rejected','post_terminal_step_rejected','hip_violation_stops_before_step','nonfinite_stops_before_step')
env_ok=all(smoke[x] for x in checks) and len(axes)==10 and all(x['steps']==100 and x['truncated'] and not x['terminated'] for x in axes)
remaining=near-first-second
reasons=[]
if remaining>0:reasons.append(f'{remaining} chassis-0 near proxy pairs remain excluded; other exclusions also exist')
if peak['normal_force_n']>0 and peak['normal_relative_jacobian_max_abs']<1e-7:reasons.append('Nonzero simulated contact force occurs along a locally constrained direction; contact/friction model needs resolution')
reasons.append('Only standing environment and short single-axis tests exist; walking objective, combined-action robustness and hardware calibration are not validated')
out={'simulation_interface_smoke_pass':env_ok,'five_axes_short_tests_pass':len(axes)==10 and all(x['steps']==100 and not x['terminated'] for x in axes),'remaining_chassis0_near_pairs':remaining,'unresolved_constrained_contact_peak_n':peak['normal_force_n'],'formal_walking_training_recommended':False,'reasons':reasons,'limitations':'Decision is a conservative project gate, not proof of real hardware safety. It intentionally distinguishes basic interface tests from walking model validity.'}
(p/'sector_training_readiness_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
