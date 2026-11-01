"""Re-create one local SB3 checkpoint ZIP from the repository's unpacked files."""
from pathlib import Path
import argparse
import zipfile

EXPECTED=('data','pytorch_variables.pth','policy.pth','policy.optimizer.pth',
          '_stable_baselines3_version','system_info.txt')
parser=argparse.ArgumentParser(description='从已展开的检查点目录恢复本地 SB3 ZIP；不会启动训练。')
parser.add_argument('checkpoint_directory',type=Path,help='包含 data、policy.pth 等文件的检查点目录')
parser.add_argument('--output',type=Path,required=True,help='本地输出 ZIP 路径，不要提交到仓库')
args=parser.parse_args()
if args.output.exists():raise SystemExit(f'输出已存在，未覆盖: {args.output}')
missing=[name for name in EXPECTED if not (args.checkpoint_directory/name).is_file()]
if missing:raise SystemExit('检查点缺少文件: '+', '.join(missing))
args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
 for name in EXPECTED:archive.write(args.checkpoint_directory/name,arcname=name)
print(f'已恢复到 {args.output}。该文件仅供本地使用，不应提交。')
