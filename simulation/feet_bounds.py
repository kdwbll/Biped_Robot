import struct,numpy as np
from pathlib import Path
p=Path(r'C:\Practice_project\biped_sim')
for f in [next((p/'meshes_original_coords').glob('*16-19-39-145-1.STL')),p/'meshes_second_leg_local'/'part6.stl']:
 with f.open('rb') as h:
  h.read(80); n=struct.unpack('<I',h.read(4))[0];v=np.frombuffer(h.read(),dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=n)['v'].reshape(-1,3)*.001
 print(f.name,'min',v.min(0),'max',v.max(0))
