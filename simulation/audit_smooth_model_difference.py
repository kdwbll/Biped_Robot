"""Verify the smooth-hip XML changes only the named joints and model label."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
p=Path(__file__).parent
a=ET.parse(p/"diagnostic_chassis0_collision_trial.xml").getroot()
b=ET.parse(p/"diagnostic_smooth_hip_trial.xml").getroot()
names={"L0_hip_candidate", "L1_hip_candidate"}
changes=[]
for x,y in zip(a.iter(),b.iter()):
 if x.tag != y.tag or len(x)!=len(y): raise AssertionError("XML structure differs")
 ax=dict(x.attrib); ay=dict(y.attrib)
 if x is a: ax.pop("model",None); ay.pop("model",None)
 if x.tag=="joint" and x.get("name") in names:
  changes.append({"joint":x.get("name"),"before":{k:ax.pop(k,None) for k in ("damping","frictionloss")},"after":{k:ay.pop(k,None) for k in ("damping","frictionloss")}})
 if ax != ay or (x.text or "").strip() != (y.text or "").strip(): raise AssertionError((x.tag,ax,ay))
if len(changes)!=2: raise AssertionError(changes)
out={"result":"only model label and two named hip joint damping/frictionloss attributes differ", "changes":changes}
(p/"smooth_hip_model_difference_audit.json").write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps(out,indent=2))
