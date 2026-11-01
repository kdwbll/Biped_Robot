"""Search tooth phasing with temporary CAD body copies; source CAD is unchanged."""
from pathlib import Path
import json
import math
import numpy as np
import pythoncom
import win32com.client as w

p=Path(__file__).parent
source=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
pythoncom.CoInitialize();sw=w.DispatchEx('SldWorks.Application.33');sw.Visible=False;doc=None
try:
 err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 doc=sw.OpenDoc6(source,2,3,'',err,warn)
 if not doc:raise RuntimeError((err.value,warn.value))
 modeler=sw.GetModeler;math_util=sw.GetMathUtility
 math_util._FlagAsMethod('CreateTransform')
 components=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 def body_for(suffix,target):
  comp=next(c for c in components if c.Name2.endswith(suffix));hits=[]
  for body in comp.GetModelDoc2.GetBodies2(0,True) or []:
   v=body.GetMassProperties(1)[3]*1e9
   if abs(v-target)<200:hits.append((body,v))
  if len(hits)!=1:raise RuntimeError((suffix,target,len(hits)))
  body=hits[0][0].Copy
  if callable(body):body=body()
  if not body.ApplyTransform(comp.Transform2):raise RuntimeError('Transform failed')
  return body
 pin=body_for('16-40-03-958-1',19219)
 gear0=body_for('16-17-29-589-1',81682)
 gear1=body_for('16-17-29-589_1-1',81622)
 def overlap(first,second):
  a=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);b=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  c=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  hit=modeler.CheckInterferenceBetweenTwoBodies(first,second,False,a,b,c)
  return sum(x.GetMassProperties(1)[3]*1e9 for x in (c.value or [])) if hit else 0.
 def rotated(original,center,angle):
  q=original.Copy
  if callable(q):q=q()
  c=math.cos(math.radians(angle));s=math.sin(math.radians(angle))
  r=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
  centre=np.array([center[0],center[1],0.]);t=centre-centre@r
  vals=list(r.reshape(-1))+list(t)+[1.,0.,0.,0.]
  x=math_util.CreateTransform(w.VARIANT(pythoncom.VT_ARRAY|pythoncom.VT_R8,vals))
  if not q.ApplyTransform(x):raise RuntimeError('rotation failed')
  return q
 self_copy=gear0.Copy
 if callable(self_copy):self_copy=self_copy()
 if overlap(gear0,self_copy)<100:raise RuntimeError('positive control failed')
 initial=[overlap(pin,gear0),overlap(gear1,gear0)]
 print('initial overlap mm3',np.round(initial,5),flush=True)
 if abs(initial[0]-9.1503)>.2 or abs(initial[1]-8.4079)>.2:raise RuntimeError('baseline disagrees with assembly interference results')
 outputs={}
 for name,moving,center,period in [('small',pin,(-.0266628795,-.006474586),360/23),
                                   ('other_large',gear1,(.042,-.073),360/50)]:
  values=[]
  for angle in np.linspace(-period/2,period/2,41):
   volume=overlap(rotated(moving,center,float(angle)),gear0)
   values.append({'angle_deg':float(angle),'overlap_mm3':volume})
  best=min(values,key=lambda x:x['overlap_mm3'])
  for angle in np.linspace(best['angle_deg']-period/40,best['angle_deg']+period/40,21):
   volume=overlap(rotated(moving,center,float(angle)),gear0)
   values.append({'angle_deg':float(angle),'overlap_mm3':volume})
  best=min(values,key=lambda x:(x['overlap_mm3'],abs(x['angle_deg'])))
  outputs[name]={'pivot_m':center,'period_degrees_assumed':period,'best':best,'samples':values}
  print(name,'best angle deg',round(best['angle_deg'],4),'overlap mm3',round(best['overlap_mm3'],6),flush=True)
 result={'source':source,'method':'Copied CAD solids rotated independently around candidate axes; saved CAD untouched.',
         'limitations':'Independent virtual gear rotation does not model attached hardware, motion constraints or tolerances.',
         'initial_overlap_mm3':initial,'search':outputs}
 (p/'cad_gear_phase_search.json').write_text(json.dumps(result,indent=2),encoding='utf8')
finally:
 if doc:sw.CloseDoc(doc.GetTitle)
 sw.ExitApp();pythoncom.CoUninitialize()
