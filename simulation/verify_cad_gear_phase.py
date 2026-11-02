"""Verify three pairwise gear overlaps at candidate angles, on CAD body copies."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import pythoncom
import win32com.client as w

p=Path(__file__).parent;phase=json.loads((p/'cad_gear_phase_search.json').read_text(encoding='utf8'))
ratio25='--25-tooth' in sys.argv
range_scan='--hypothetical-range' in sys.argv
ratio=.5 if ratio25 else .46
tooth_count=25 if ratio25 else 23
attach='--attach' in sys.argv
pythoncom.CoInitialize();sw=w.GetActiveObject('SldWorks.Application.33') if attach else w.DispatchEx('SldWorks.Application.33')
if not attach:sw.Visible=False
doc=None;opened_here=False
try:
 err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 assembly=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
 doc=sw.GetOpenDocumentByName(assembly) if attach else None
 if not doc:
  doc=sw.OpenDoc6(assembly,2,3,'',err,warn);opened_here=bool(doc)
 if not doc:raise RuntimeError((err.value,warn.value))
 modeler=sw.GetModeler;util=sw.GetMathUtility;util._FlagAsMethod('CreateTransform')
 cs=[x for x in doc.GetComponents(False) if x.GetPathName.lower().endswith('.sldprt') and not x.IsSuppressed]
 def find(suffix,volume):
  comp=next(c for c in cs if c.Name2.endswith(suffix))
  bodies=[b for b in comp.GetModelDoc2.GetBodies2(0,True) or [] if abs(b.GetMassProperties(1)[3]*1e9-volume)<200]
  if len(bodies)!=1:raise RuntimeError((suffix,len(bodies)))
  body=bodies[0].Copy
  if callable(body):body=body()
  if not body.ApplyTransform(comp.Transform2):raise RuntimeError('Transform failed')
  return body
 def rotate(body,center,angle):
  a=math.radians(angle);c=math.cos(a);s=math.sin(a);R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
  x=np.array([*center,0.]);t=x-x@R;data=list(R.reshape(-1))+list(t)+[1.,0.,0.,0.]
  b=body.Copy
  if callable(b):b=b()
  if not b.ApplyTransform(util.CreateTransform(w.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R8,data))):raise RuntimeError('rotation failed')
  return b
 def overlap(a,b):
  x=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);y=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  z=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  return sum(v.GetMassProperties(1)[3]*1e9 for v in (z.value or [])) if modeler.CheckInterferenceBetweenTwoBodies(a,b,False,x,y,z) else 0.
 pin=find('16-40-03-958-1',19219);left=find('16-17-29-589-1',81682);right=find('16-17-29-589_1-1',81622)
 copy=left.Copy
 if callable(copy):copy=copy()
 if overlap(left,copy)<100:raise RuntimeError('positive overlap control failed')
 small=phase['search']['small'];large=phase['search']['other_large']
 small_base=small['best']['angle_deg'];large_base=large['best']['angle_deg']
 pin_at=rotate(pin,small['pivot_m'],small_base);right_at=rotate(right,large['pivot_m'],large_base)
 out={'pinion_first_big_mm3':overlap(pin_at,left),'first_big_second_big_mm3':overlap(left,right_at),
      'pinion_second_big_mm3':overlap(pin_at,right_at),'cycle_samples':[],
      'assumed_tooth_count_pinion':tooth_count,'assumed_ratio':ratio,
      'note':'Exact CAD body copies, 17 samples across one assumed pinion tooth period; not continuous clearance or full-cycle proof.'}
 samples = np.linspace(-10., 10., 41) if range_scan else np.linspace(0,360/tooth_count,17)
 if range_scan:
  out['note'] = '41 discrete samples over hypothetical pinion range -10 to +10 degrees (large gears +/-5 degrees); not a measured physical limit or continuous collision proof.'
 for delta in samples:
  pin_d=rotate(pin,small['pivot_m'],small_base+float(delta))
  left_d=rotate(left,(.042,.032),-ratio*float(delta))
  right_d=rotate(right,large['pivot_m'],large_base+ratio*float(delta))
  row={'pinion_delta_deg':float(delta),'pinion_first_big_mm3':overlap(pin_d,left_d),
       'first_big_second_big_mm3':overlap(left_d,right_d),'pinion_second_big_mm3':overlap(pin_d,right_d)}
  out['cycle_samples'].append(row)
  print('pinion delta deg',round(delta,3),'interference mm3',
        round(row['pinion_first_big_mm3'],5),round(row['first_big_second_big_mm3'],5),flush=True)
 filename = 'cad_gear_hypothetical_range_audit.json' if range_scan else ('cad_gear_phase_25_tooth_verification.json' if ratio25 else 'cad_gear_phase_pairwise_verification.json')
 (p/filename).write_text(json.dumps(out,indent=2),encoding='utf8')
 print('maximum sampled pairwise overlap mm3',max(max(x[k] for k in ('pinion_first_big_mm3','first_big_second_big_mm3','pinion_second_big_mm3')) for x in out['cycle_samples']),flush=True)
finally:
 if doc and (not attach or opened_here):sw.CloseDoc(doc.GetTitle)
 if not attach:sw.ExitApp()
 pythoncom.CoUninitialize()
