"""Check saved-pose interference between selected copied CAD bodies, read-only."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

p=Path(__file__).parent
path=r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
pythoncom.CoInitialize();sw=w.DispatchEx('SldWorks.Application.33');sw.Visible=False;doc=None
try:
 err=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0);warn=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
 doc=sw.OpenDoc6(path,2,3,'',err,warn)
 if not doc:raise RuntimeError((err.value,warn.value))
 modeler=sw.GetModeler
 cs=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 torso=next(c for c in cs if c.Name2.endswith('16-40-03-958-1'))
 legs=[c for c in cs if c.Name2.endswith(('16-17-29-589-1','16-17-29-589_1-1'))]
 def selected(comp,targets):
  out=[]
  for body in comp.GetModelDoc2.GetBodies2(0,True) or []:
   vol=body.GetMassProperties(1)[3]*1e9
   for expected in targets:
    if abs(vol-expected)<200:
     new=body.Copy
     if callable(new):new=new()
     if not new.ApplyTransform(comp.Transform2):raise RuntimeError('Copy transform failed')
     out.append((round(vol,3),body.Name,new));break
  return out
 chassis=selected(torso,[135649,31826,34786,43915,143769])
 # A body checked against its own independent copy must produce an overlap;
 # without this positive control, a false return is not a reliable no-clash result.
 va=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
 vb=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
 vc=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
 checkcopy=chassis[0][2].Copy
 if callable(checkcopy):checkcopy=checkcopy()
 positive=modeler.CheckInterferenceBetweenTwoBodies(chassis[0][2],checkcopy,False,va,vb,vc)
 print('positive control identical body result',bool(positive),'intersection count',len(vc.value or []),flush=True)
 if not positive:
  raise RuntimeError('Positive control failed; body-level collision results are inconclusive')
 rows=[]
 for leg in legs:
  towers=selected(leg,[129952])
  for tv,tn,tb in towers:
   for cv,cn,cb in chassis:
    a=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
    b=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
    result=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
    row={'leg':leg.Name2,'tower_volume_mm3':tv,'tower_body':tn,'chassis_volume_mm3':cv,'chassis_body':cn}
    try:
     success=modeler.CheckInterferenceBetweenTwoBodies(cb,tb,False,a,b,result)
     bodies=result.value or []
     row['success']=bool(success);row['interference_body_count']=len(bodies)
     row['interference_volume_mm3']=sum(float(x.GetMassProperties(1)[3])*1e9 for x in bodies)
    except Exception as exc:row['error']=str(exc)
    rows.append(row)
    print(leg.Name2.split('/')[-1],'chassis mm3',cv,'result',row.get('interference_volume_mm3'),row.get('error',''),flush=True)
 (p/'sw_target_bodies_interference_saved_pose.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
finally:
 if doc:sw.CloseDoc(doc.GetTitle)
 sw.ExitApp();pythoncom.CoUninitialize()
