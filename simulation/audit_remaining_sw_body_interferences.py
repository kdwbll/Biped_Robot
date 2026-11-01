"""Identify individual solid bodies in saved-pose CAD interferences, read-only."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

p=Path(__file__).parent
path=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
pythoncom.CoInitialize();sw=w.DispatchEx('SldWorks.Application.33');sw.Visible=False;doc=None
try:
 e=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 doc=sw.OpenDoc6(path,2,3,'',e,warn)
 if not doc:raise RuntimeError((e.value,warn.value))
 modeler=sw.GetModeler
 components=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 torso=next(c for c in components if c.Name2.endswith('16-40-03-958-1'))
 leg0=next(c for c in components if c.Name2.endswith('16-17-29-589-1'))
 leg1=next(c for c in components if c.Name2.endswith('16-17-29-589_1-1'))
 def load(comp):
  out=[]
  for b in comp.GetModelDoc2.GetBodies2(0,True) or []:
   clone=b.Copy
   if callable(clone):clone=clone()
   if not clone.ApplyTransform(comp.Transform2):raise RuntimeError('Cannot transform copied body')
   box=list(clone.GetBodyBox());out.append({'body':clone,'name':b.Name,'volume_mm3':float(b.GetMassProperties(1)[3]*1e9),
    'bounding_box_assembly_mm':[x*1000 for x in box]})
  return out
 T,A,B=map(load,(torso,leg0,leg1))
 def calculate(left,right):
  a=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  b=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  c=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  hit=modeler.CheckInterferenceBetweenTwoBodies(left,right,False,a,b,c)
  return bool(hit),list(c.value or [])
 copy=A[0]['body'].Copy
 if callable(copy):copy=copy()
 if not calculate(A[0]['body'],copy)[0]:raise RuntimeError('positive overlap control failed')
 results=[]
 for group,left,right in [('torso_leg0',T,A),('leg0_leg1',A,B)]:
  for a in left:
   for b in right:
    hit,volumes=calculate(a['body'],b['body'])
    if not hit:continue
    vb=[{'volume_mm3':float(x.GetMassProperties(1)[3]*1e9),
         'box_assembly_mm':[v*1000 for v in x.GetBodyBox()]} for x in volumes]
    row={'group':group,'first':{k:v for k,v in a.items() if k!='body'},
         'second':{k:v for k,v in b.items() if k!='body'},'intersections':vb}
    results.append(row)
    print(group,a['name'],round(a['volume_mm3'],1),b['name'],round(b['volume_mm3'],1),
          'overlap volumes mm3',[round(z['volume_mm3'],5) for z in vb],flush=True)
 (p/'sw_remaining_body_interferences.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print('body pairs with detected interference',len(results),'sizes',len(T),len(A),len(B),flush=True)
finally:
 if doc:sw.CloseDoc(doc.GetTitle)
 sw.ExitApp();pythoncom.CoUninitialize()
