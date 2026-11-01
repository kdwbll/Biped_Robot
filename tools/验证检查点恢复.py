"""Check local checkpoint reconstruction byte-for-byte without training."""
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import zipfile

root=Path(__file__).resolve().parent.parent
source=root/'simulation'/'模型检查点'/'continued_gpu_98304'
target=Path(tempfile.gettempdir())/f'biped-check-{uuid.uuid4().hex}.zip'
try:
 subprocess.run([sys.executable,str(root/'tools'/'恢复模型检查点.py'),str(source),'--output',str(target)],check=True)
 with zipfile.ZipFile(target) as z:
  names=set(z.namelist());expected={f.name for f in source.iterdir() if f.is_file()}
  if names!=expected:raise RuntimeError('Checkpoint member list differs')
  for name in names:
   if z.read(name)!=(source/name).read_bytes():raise RuntimeError(f'Checkpoint member differs: {name}')
 print(f'检查点 {len(names)} 个文件逐字节一致；未运行训练。')
finally:
 if target.is_file():target.unlink()
