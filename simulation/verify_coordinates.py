import json,struct
from pathlib import Path
import numpy as np
p=Path(__file__).parent
for a in json.loads((p/'sw_boxes.json').read_text())[:3]:
 f=next(f for f in (p/'meshes_original_coords').glob('*.STL') if f.stem.endswith(a['name']))
 with f.open('rb') as h:
  h.read(80); n=struct.unpack('<I',h.read(4))[0]; v=np.frombuffer(h.read(),dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=n)['v'].reshape(-1,3)*0.001
 print(a['name'], 'min',np.round(v.min(0),4),'SW approx',np.round(a['box_m'][:3],4))
