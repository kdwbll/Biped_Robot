"""Check suspect torso/tower CAD solid pairs at saved and trial hip poses."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import pythoncom
import win32com.client as w

p=Path(__file__).parent
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved assembly in SolidWorks 2025')
 components=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 modeler=sw.GetModeler;util=sw.GetMathUtility;util._FlagAsMethod('CreateTransform')
 def body(suffix,volume):
  comp=next(c for c in components if c.Name2.endswith(suffix))
  match=[b for b in (comp.GetModelDoc2.GetBodies2(0,True) or []) if abs(b.GetMassProperties(1)[3]*1e9-volume)<200]
  if len(match)!=1:raise RuntimeError((suffix,volume,len(match)))
  b=match[0].Copy
  if callable(b):b=b()
  if not b.ApplyTransform(comp.Transform2):raise RuntimeError('Cannot transform copied body')
  return b
 def rotate(b,center,angle):
  a=math.radians(angle);c=math.cos(a);s=math.sin(a);R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
  x=np.array([*center,0.]);t=x-x@R
  tr=util.CreateTransform(w.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R8,list(R.reshape(-1))+list(t)+[1.,0.,0.,0.]))
  result=b.Copy
  if callable(result):result=result()
  if not result.ApplyTransform(tr):raise RuntimeError('Cannot rotate copied body')
  return result
 def overlap(a,b):
  x=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);y=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);z=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  hit=bool(modeler.CheckInterferenceBetweenTwoBodies(a,b,False,x,y,z))
  return {'intersects':hit,'volume_mm3':sum(float(v.GetMassProperties(1)[3]*1e9) for v in (z.value or []))}
 torso={0:body('16-40-03-958-1',135648.8),5:body('16-40-03-958-1',143768.7)}
 legs={0:body('16-17-29-589-1',129952.3),1:body('16-17-29-589_1-1',129952.3)}
 if overlap(torso[5],torso[5])['volume_mm3']<100:raise RuntimeError('Positive control failed')
 phase=json.loads((p/'cad_gear_phase_search.json').read_text(encoding='utf8'))
 pivot=phase['search']['other_large']['pivot_m'];phase_deg=phase['search']['other_large']['best']['angle_deg']
 rows=[];range_scan='--range' in sys.argv
 if range_scan:
  for delta in np.linspace(-5.,5.,11):
   for ci in (0,5):
    for li in (0,1):
     angle=(-float(delta)) if li==0 else phase_deg+float(delta)
     center=(.042,.032) if li==0 else pivot
     row={'chassis_source_index':ci,'leg':li,'drive_delta_deg':float(delta),'leg_z_angle_deg':angle,
          **overlap(torso[ci],rotate(legs[li],center,angle))}
     rows.append(row);print(row,flush=True)
 else:
  for ci in (0,5):
   for li in (0,1):
    for angle in ([0.,phase_deg] if li==1 else [0.]):
     candidate=rotate(legs[li],pivot,angle) if angle else legs[li]
     rows.append({'chassis_source_index':ci,'leg':li,'leg_z_angle_deg':angle,**overlap(torso[ci],candidate)})
 out={'results':rows,'note':'Exact CAD solid copies. Sampled hypothetical hip angles only; this is not continuous or whole-assembly collision validation.'}
 (p/('cad_collision_source_range.json' if range_scan else 'cad_collision_source_interference.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
 print('samples',len(rows),'intersecting',sum(x['intersects'] for x in rows),flush=True)
finally:pythoncom.CoUninitialize()
