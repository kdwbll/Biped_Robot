import struct,json,numpy as np
from pathlib import Path
p=Path(__file__).parent; items=json.loads((p/'sw_boxes.json').read_text()); offsets=[]
for f in (p/'meshes').glob('*.STL'):
 matches=[x for x in items if f.stem.endswith(x['name'])]
 if len(matches)!=1: print('MATCH ERROR',f.name,len(matches)); continue
 with f.open('rb') as h:
  h.read(80); n=struct.unpack('<I',h.read(4))[0]; v=np.frombuffer(h.read(),dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=n)['v'].reshape(-1,3)*0.001
 box=np.array(matches[0]['box_m']); mid=(v.min(0)+v.max(0))/2; sw=(box[:3]+box[3:])/2; offsets.append(mid-sw)
offsets=np.array(offsets); med=np.median(offsets,axis=0)
print('parts',len(offsets),'estimated export offset m',med,'max midpoint deviation mm',np.max(np.abs(offsets-med),axis=0)*1000)
