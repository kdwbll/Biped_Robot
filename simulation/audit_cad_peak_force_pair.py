"""Check peak-force chassis body and leg solid in SolidWorks over sampled hip angles."""
from pathlib import Path
import json,math
import numpy as np
import pythoncom
import win32com.client as w
p=Path(__file__).parent;pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33');doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open SolidWorks assembly')
 comps=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 def bodies(suffix,volume):
  c=next(c for c in comps if c.Name2.endswith(suffix));out=[]
  for b in c.GetModelDoc2.GetBodies2(0,True) or []:
   if abs(b.GetMassProperties(1)[3]*1e9-volume)>200:continue
   cp=b.Copy
   if callable(cp):cp=cp()
   if not cp.ApplyTransform(c.Transform2):raise RuntimeError('Cannot transform copy')
   out.append(cp)
  return out
 candidates=bodies('16-40-03-958-1',31825.86);chassis=next(b for b in candidates if b.GetBodyBox()[1]>.01)
 legs=bodies('16-17-29-589-1',129952.3)
 if len(legs)!=1:raise RuntimeError('Leg body not unique')
 leg=legs[0];modeler=sw.GetModeler;util=sw.GetMathUtility;util._FlagAsMethod('CreateTransform')
 def collision(a,b):
  x=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);y=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);z=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  hit=bool(modeler.CheckInterferenceBetweenTwoBodies(a,b,False,x,y,z));return {'intersects':hit,'overlap_volume_mm3':sum(float(q.GetMassProperties(1)[3]*1e9) for q in (z.value or []))}
 if collision(chassis,chassis)['overlap_volume_mm3']<100:raise RuntimeError('Interference positive control failed')
 rows=[]
 for angle in np.linspace(-5,5,11):
  t=math.radians(angle);c=math.cos(t);s=math.sin(t);R=np.array([[c,s,0],[-s,c,0],[0,0,1.]]);origin=np.array([.042,.032,0]);translation=origin-origin@R
  transform=util.CreateTransform(w.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R8,list(R.reshape(-1))+list(translation)+[1.,0.,0.,0.]))
  cp=leg.Copy
  if callable(cp):cp=cp()
  if not cp.ApplyTransform(transform):raise RuntimeError('Rotation failed')
  row={'cad_leg_rotation_deg':float(angle),**collision(chassis,cp)};rows.append(row);print(row,flush=True)
 out={'model_pair':['dc_chassis_1_1','tower_L0_tower_8'],'chassis_box_mm':[x*1000 for x in chassis.GetBodyBox()],'tower_box_mm':[x*1000 for x in leg.GetBodyBox()],'results':rows,'limitations':'Only these two nominal CAD solids at 11 hypothetical rotation poses. Zero intersection does not establish clearance or real bearing load. CAD and simulated leg phase may differ; other parts are unchecked.'}
 (p/'cad_peak_force_pair_sweep.json').write_text(json.dumps(out,indent=2),encoding='utf8')
finally:pythoncom.CoUninitialize()
