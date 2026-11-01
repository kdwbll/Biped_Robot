"""Read exact initial-pose body distances in the existing SolidWorks assembly.

Requires a user-opened assembly. Does not select entities, edit, save, or close it.
"""
from pathlib import Path
import json
import math
import pythoncom
import win32com.client as w

p=Path(__file__).parent
pythoncom.CoInitialize()
try:
 sw=w.GetActiveObject('SldWorks.Application.33')
 doc=sw.GetOpenDocumentByName(r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM')
 if not doc:raise RuntimeError('Open the saved SimpleBiped2a_new.SLDASM in SolidWorks 2025')
 comps=[c for c in doc.GetComponents(False) if c.GetPathName.lower().endswith('.sldprt') and not c.IsSuppressed]
 def get_body(suffix,volume):
  comp=next(c for c in comps if c.Name2.endswith(suffix))
  info=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  found=[b for b in (comp.GetBodies3(0,info) or []) if abs(b.GetMassProperties(1)[3]*1e9-volume)<200]
  if len(found)!=1:raise RuntimeError((suffix,volume,len(found)))
  return found[0]
 torso={0:get_body('16-40-03-958-1',135648.8),5:get_body('16-40-03-958-1',143768.7)}
 legs={0:get_body('16-17-29-589-1',129952.3),1:get_body('16-17-29-589_1-1',129952.3)}
 def distance(a,b):
  p1=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None);p2=w.VARIANT(pythoncom.VT_BYREF|pythoncom.VT_VARIANT,None)
  result=float(doc.ClosestDistance(a,b,p1,p2))
  if not math.isfinite(result) or result<0:raise RuntimeError(f'ClosestDistance returned invalid result {result}')
  return {'distance_mm':result*1000,'point1_m':list(p1.value) if p1.value else None,'point2_m':list(p2.value) if p2.value else None}
 zero=distance(torso[0],torso[0])
 if zero['distance_mm']>1e-5:raise RuntimeError('Identical-body distance control failed')
 rows=[]
 for ci in (0,5):
  for leg in (0,1):
   row={'chassis_source_index':ci,'leg':leg,**distance(torso[ci],legs[leg])};rows.append(row);print(row,flush=True)
 out={'saved_assembly':doc.GetPathName,'initial_pose_only':True,'same_body_control':zero,'results':rows,
  'limitations':'SolidWorks exact minimum distance at current assembly pose only. Zero distance does not distinguish touching from intersection; temporary bodies and hip rotation are not measured.'}
 (p/'solidworks_closest_distance_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
finally:pythoncom.CoUninitialize()
