import struct,glob,numpy as np,os
for p in glob.glob(r'C:\Practice_project\biped_sim\meshes\*.STL'):
 with open(p,'rb') as f:
  f.read(80); n=struct.unpack('<I',f.read(4))[0]; a=np.frombuffer(f.read(),dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),count=n)['v'].reshape(-1,3)
  print(os.path.basename(p)[-45:], 'min',np.round(a.min(0),1),'max',np.round(a.max(0),1))
