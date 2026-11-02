"""Verify the coverage and results of the saved dense SolidWorks body sweep."""
from pathlib import Path
import json
p=Path(__file__).parent;d=json.loads((p/'cad_collision_source_range_dense.json').read_text(encoding='utf8'));rows=d['results']
if len(rows)!=404:raise RuntimeError('Expected 404 CAD checks')
expected={round(-5+i*.1,6) for i in range(101)}
for chassis in (0,5):
 for leg in (0,1):
  subset=[r for r in rows if r['chassis_source_index']==chassis and r['leg']==leg]
  if len(subset)!=101 or {round(r['drive_delta_deg'],6) for r in subset}!=expected:raise RuntimeError('Incomplete angle grid')
intersections=[r for r in rows if r['intersects'] or r['volume_mm3']>0]
out={'source_checks':len(rows),'groups':4,'angles_per_group':101,'intersections':len(intersections),'limitations':'Validates saved coverage/results, not CAD execution or continuous angles. Nominal CAD solids and selected body pairs only; real tolerances and friction not covered.'}
(p/'cad_collision_dense_coverage_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
