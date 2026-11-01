"""Generate a per-file Chinese inventory without third-party dependencies."""
from pathlib import Path
from urllib.parse import quote
import ast
import json
import re

ROOT=Path(__file__).resolve().parent.parent
INDEX=ROOT/'文件清单.md'
SKIP={'.git','.venv','.pio','__pycache__','warp_cache','gpu_packages','warp_test_packages'}
ARCHIVES={'.zip','.7z','.rar','.tar','.gz','.tgz','.bz2','.xz'}
WORDS={'gear':'齿轮','gears':'齿轮','servo':'舵机','shaft':'轴','pinion':'小齿轮','hip':'髋部',
       'knee':'膝部','leg':'腿部','legs':'双腿','tower':'塔架','chassis':'机身','body':'刚体',
       'standing':'站姿','stand':'站立','pose':'姿态','neutral':'中立姿态','collision':'碰撞',
       'contact':'接触','contacts':'接触','forces':'力','force':'力','mass':'质量','inertia':'惯量',
       'phase':'相位','teeth':'齿数','tooth':'齿数','ratio':'传动比','drive':'驱动','drives':'驱动',
       'kinematic':'运动学','kinematics':'运动学','rigidity':'刚性连接','visual':'视觉网格',
       'mesh':'网格','meshes':'网格','refined':'精细代理','candidate':'候选','diagnostic':'诊断',
       'constraints':'约束','constraint':'约束','joints':'关节','limit':'限位','limits':'限位',
       'floor':'地面','ground':'地面','friction':'摩擦','support':'辅助支撑','supported':'辅助支撑',
       'weight':'重量','stability':'稳定性','sensitivity':'敏感性','timestep':'步长','symmetric':'对称',
       'symmetry':'对称性','mirrored':'镜像','preview':'预览','viewer':'查看器','original':'原始',
       'current':'当前','after':'修改后','before':'修改前','fix':'修复','post':'修改后',
       'cad':'CAD','sw':'SolidWorks','mujoco':'MuJoCo','warp':'Warp','gpu':'GPU','cpu':'CPU',
       'output':'输出','output':'输出','data':'数据','record':'记录','state':'状态','progress':'进度',
       'training':'训练','train':'训练','checkpoint':'检查点','policy':'策略','ppo':'PPO',
       'walking':'行走','gait':'步态','step':'步进','stepping':'踏步','action':'动作',
       'test':'测试','estimated':'估算','theoretical':'理论','interference':'干涉','interferences':'干涉',
       'bounds':'边界','boxes':'包围盒','axes':'轴线','axis':'轴线','geometry':'几何',
       'robot':'机器人','model':'模型','models':'模型','baseline':'基线','motion':'运动',
       'interference':'干涉','numerical':'数值','solver':'求解器','fidelity':'精度'}

def topic(name):
 parts=[w for w in re.split(r'[_\-.]+',name) if w]
 return '、'.join(WORDS.get(w.lower(),w) for w in parts[:12])

def text_title(path):
 try:
  for line in path.read_text(encoding='utf-8-sig').splitlines():
   if line.startswith('# '):return line[2:].strip()[:80]
 except (OSError,UnicodeError):pass
 return path.stem

def python_info(path):
 name=path.stem;low=name.lower()
 custom={
 'candidate_standing_servo':('假设的带积分舵机站姿控制器','由站姿诊断或预览脚本导入；增益是仿真假设。'),
 'theoretical_servo':('按商品参数假设的舵机模型与限矩','由仿真脚本导入；不能作为实物驱动。'),
 'stabilized_servo':('第五轴阻尼调整的旧模拟舵机候选','由旧站姿诊断脚本导入对照。'),
 'view_gear_kinematic_preview':('25:50 齿轮带动腿架的纯运动学动画','运行同目录“查看齿轮腿部联动演示.bat”；不计算动力学或训练。'),
 'view_standing_candidate':('假设舵机控制下的站姿可视化','运行同目录“查看站姿控制候选.bat”；不是实物控制。'),
 'restore_checkpoint':('从展开文件恢复模型检查点','按 README 的检查点恢复步骤使用。'),
 '恢复模型检查点':('从展开文件恢复本地 SB3 检查点','运行 `python tools/恢复模型检查点.py --help` 查看参数；不启动训练。'),
 '生成文件索引':('按仓库实际文件生成本清单','在仓库根目录运行 `python tools/生成文件索引.py`。'),
 }
 if name in custom:return custom[name]
 try:
  tree=ast.parse(path.read_text(encoding='utf-8-sig'))
  doc=ast.get_docstring(tree)
  executable=any(isinstance(n,(ast.For,ast.While,ast.With,ast.Try,ast.If)) or
                 isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) or
                 isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) for n in tree.body)
 except (SyntaxError,UnicodeError,OSError):doc=None;executable=True
 if low.startswith('train'):
  return (f'历史训练或课程训练脚本 `{name}`','仅在用户明确要求并检查环境后运行；可能产生模型、日志或长时间占用算力。')
 if low.startswith(('view','render','show')):
  category='可视化或渲染'
 elif low.startswith(('audit','verify','check','test','probe','diagnose','validate')):
  category='检查或诊断'
 elif low.startswith(('build','make','save','apply','attach','correct','filter','split','decompose','refine','solve')):
  category='构建或修改实验候选'
 elif low.startswith(('compare','summarize','analyze','benchmark','evaluate','eval','extract','export','list','match')):
  category='比较、提取或分析'
 else:category='仿真或项目辅助'
 subject=re.sub(r'^(audit|verify|check|test|probe|diagnose|validate|build|make|save|apply|attach|correct|filter|split|decompose|refine|solve|compare|summarize|analyze|benchmark|evaluate|eval|extract|export|list|match|view|render|show)_?','',name,flags=re.I)
 purpose=f'{topic(subject or name)}：{category}脚本'
 if doc:
  summary=' '.join(doc.split())[:140].replace('|','/')
  purpose+=f'；脚本自述：{summary}'
 if not executable:return purpose,'供其他 Python 程序导入；先阅读类和函数的参数。'
 return purpose,'在该文件所在目录用 `python "'+path.name+'"` 启动前，先检查依赖、参数、绝对路径和写入输出；部分旧脚本只适用于原电脑。'

def describe(path,rel):
 name=path.name;ext=path.suffix.lower();parts=rel.parts
 if '模型检查点' in parts:
  data={'data':('SB3 检查点元数据和算法配置','由恢复后的 SB3 模型读取；不要手工编辑。'),
        'policy.pth':('神经网络策略参数','由 SB3 加载；不能证明该策略可控制实物。'),
        'policy.optimizer.pth':('策略优化器状态','用于恢复对应训练检查点；不要单独当策略运行。'),
        'pytorch_variables.pth':('额外 PyTorch 模型变量','由对应检查点加载流程读取。'),
        '_stable_baselines3_version':('保存检查点时的 SB3 版本','文本查看并用于排查版本兼容。'),
        'system_info.txt':('保存检查点的运行环境信息','文本查看并用于排查环境兼容。')}
  if name in data:
   purpose,use=data[name];return purpose,use+' 该目录可用 README 中的恢复工具重建本地检查点。'
 if ext=='.py':return python_info(path)
 if ext=='.md':return '文档：'+text_title(path),'用 Markdown 阅读器打开；历史文档需与 README 的最新状态对照。'
 if name=='.gitignore':return 'Git 忽略规则，排除缓存和通用压缩包','Git 自动读取；新增类型时维护规则。'
 if name=='.gitattributes':return 'Git 中文件换行和二进制处理规则','Git 自动读取，修改前检查 CAD 和模型文件完整性。'
 if ext in ('.sldasm','.sldprt'):
  return ('SolidWorks '+('装配体' if ext=='.sldasm' else '零件')+'：'+path.stem,
          '用 SolidWorks 打开；装配引用缺失时选择同目录的对应零件。')
 if ext in ('.step','.stp'):
  return 'CAD STEP 交换文件：'+path.stem,'用 CAD 软件或切片软件导入，打印前核对比例和结构。'
 if ext=='.3mf':return '3MF 打印工程：'+path.stem,'用兼容的切片软件打开，核对设备、材料和打印参数。'
 if ext=='.stl':
  folder=path.parent.name
  meaning=('精细塔架凸碰撞代理' if 'refined' in folder else '碰撞代理网格' if any(x in folder for x in ('collision','decomposed')) else
           '齿轮相位预览网格' if folder=='gear_phase_visuals' else '轴长修改后的 CAD 视觉导出' if folder=='meshes_after_shaft_fix' else 'CAD 或模型视觉网格')
  return meaning+'：'+path.stem,'由引用该文件的 MuJoCo XML 使用；可用网格查看器单独检查，单位和坐标系需按模型确认。'
 if ext=='.xml':
  try:
   import xml.etree.ElementTree as ET
   root=ET.parse(path).getroot();model=root.get('model') or path.stem
  except Exception:model=path.stem
  return 'MuJoCo 模型或诊断配置：'+model,'由同目录 Python 脚本加载；直接打开 XML 不会自动运行舵机控制器。'
 if ext=='.json':
  try:
   obj=json.loads(path.read_text(encoding='utf-8-sig'));keys=list(obj)[:4] if isinstance(obj,dict) else []
  except Exception:keys=[]
  return '结构化记录或诊断结果：'+topic(path.stem)+('；字段示例：'+', '.join(map(str,keys)) if keys else ''),'用文本编辑器查看；其数值需结合对应同名脚本和核验报告理解。'
 if ext=='.jsonl':return '逐行 JSON 实验记录：'+path.stem,'每行是一个记录，用文本工具分析；不表示已验证策略。'
 if ext=='.log':return '历史运行日志：'+path.stem,'用文本编辑器查看对应实验的输出或错误。'
 if ext=='.npz':return 'NumPy 数值数组或初始姿态数据：'+path.stem,'供相关 Python 脚本使用；用 `numpy.load` 查看数组名称。'
 if ext=='.pth':return 'PyTorch 参数或检查点数据：'+path.stem,'由对应 PyTorch/SB3 程序加载；与所属模型版本一起使用。'
 if ext in ('.png','.jpg','.jpeg'):return '模型或实验预览图：'+path.stem,'用图片查看器打开；静态图片不能验证动力学。'
 if ext=='.bat':return 'Windows 启动入口：'+path.stem,'确认依赖和脚本用途后双击；训练入口可能长时间占用资源。'
 if ext=='.ino':return 'Arduino 源码：'+path.stem,'在对应 Arduino/PlatformIO 工程中阅读和构建；硬件参数需实测。'
 if ext in ('.cpp','.h','.hpp'):return 'C/C++ 源代码或头文件：'+path.stem,'由所属 PlatformIO 工程构建，或供其他源文件包含。'
 if ext=='.ini':return '项目或依赖配置：'+path.stem,'用文本编辑器查看，供所属工具或工程读取。'
 if ext=='.txt':return '说明、依赖列表或文本结果：'+path.stem,'用文本编辑器查看；依赖列表可按 README 安装。'
 if name.lower()=='license':return '原开源项目许可证','阅读许可证及其中的使用和再分发条件。'
 return '项目文件：'+name,'按照所属目录与同目录文档使用；不确定时先查看文件内容。'

files=sorted((f for f in ROOT.rglob('*') if f.is_file() and not any(s in SKIP for s in f.relative_to(ROOT).parts)
              and f.suffix.lower() not in ARCHIVES),key=lambda x:x.relative_to(ROOT).as_posix().casefold())
if INDEX not in files:files.append(INDEX);files.sort(key=lambda x:x.relative_to(ROOT).as_posix().casefold())
lines=['# 项目完整文件清单','','本清单逐项说明仓库文件的用途与使用方法。历史实验结果不代表实物验证。',
       '路径可点击；Windows 路径或生成脚本输出请先检查。','',f'共 {len(files)} 个文件。','',
       '| 文件 | 用途 | 如何使用 |','| --- | --- | --- |']
for path in files:
 rel=path.relative_to(ROOT);label=rel.as_posix().replace('|','\\|');url=quote(rel.as_posix(),safe='/')
 if path==INDEX:purpose,use='本逐文件索引','查阅每个仓库文件的用途和使用方式；由索引生成脚本更新。'
 else:purpose,use=describe(path,rel)
 lines.append(f'| [{label}]({url}) | {purpose.replace(chr(124), "/")} | {use.replace(chr(124), "/")} |')
INDEX.write_text('\n'.join(lines)+'\n',encoding='utf8')
print(f'已为 {len(files)} 个文件生成 {INDEX}')
