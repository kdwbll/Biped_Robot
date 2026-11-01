# Biped Robot：双足机器人复刻与仿真实验

本项目基于 [XRobots / YouCanBuildBiPed](https://github.com/XRobots/YouCanBuildBiPed)，保存 SolidWorks 装配、打印资料、ESP32 工程、MuJoCo 模型、实验脚本和历史结果。原项目快照及其许可证保留在 [`upstream/`](upstream/)。

**项目仍在核验阶段：没有经过实物标定，也没有验证可靠行走。** 本仓库中的站立结果、模型权重、材料和齿轮参数是实验或假设，不能直接作为实物控制参数。当前不自动启动训练。

## 目录与文件说明

| 路径 | 内容及用途 |
| --- | --- |
| [`cad/`](cad/) | 用户当前的 SolidWorks 装配和零件；主装配为 `SimpleBiped2a_new.SLDASM`。 |
| [`printing/`](printing/) | STEP 和 3MF 打印相关文件；打印前应核对尺寸、数量和材料。 |
| [`firmware/`](firmware/) | PlatformIO 主工程、ESP32 和传感器测试工程、调试说明。 |
| [`simulation/`](simulation/) | 模型、网格、脚本、结果、可视化入口和已展开的历史模型检查点。 |
| [`upstream/`](upstream/) | 原作者公开项目快照，包括原始设计、代码和许可证。 |
| [`tools/`](tools/) | 文件索引生成及本地恢复模型检查点的工具。 |
| [`项目交接记录.md`](项目交接记录.md) | 项目背景、过去的硬件选择与未决事项；部分信息是历史记录。 |
| [`整理记录.json`](整理记录.json) | 本次整理来源、排除的缓存及各检查点原路径和展开目录。 |
| [`文件清单.md`](文件清单.md) | **逐个文件的用途和使用方法**；文件很多，可在页面内搜索文件名。 |

所有文件的逐项说明见 [完整文件清单](文件清单.md)。仓库的后续更新要求见 [AGENTS.md](AGENTS.md)。

## 当前模型进度

- 用户已将一侧上部舵机轴缩短约 1 mm；当前 CAD 复核的装配干涉从 7 处降至 6 处，剩余干涉出现在齿轮啮合处。
- CAD 网格周期和有限角度的实体干涉采样更支持小齿轮 **25 齿**、大齿轮 **50 齿**；早期 23:50 是旧假设。该结论尚需原设计或实物核实。
- 两侧大齿轮分别与对应腿架一起转动，小齿轮跟随中央舵机轴。隔离运动学测试中，小齿轮约转 1.35°，两侧腿架绕共同轴分别约转 -0.675° 和 +0.675°。
- 最新实验模型是 [`diagnostic_post_shaft_fix_gear_phase_25.xml`](simulation/diagnostic_post_shaft_fix_gear_phase_25.xml)。其中 ±5° 髋部范围、舵机控制增益、质量与齿轮相位仍是假设；齿轮使用理想耦合，没有完整的齿面接触。还存在尚未逐一核对的碰撞对，带地面转向时仍有目标误差。
- 已做不启动训练的 20 秒中立站姿、有限幅度推力和步长对照；这只说明所测数值场景的表现。完整证据、旧结果及限制见 [`核验进展_2026-11-01.md`](simulation/核验进展_2026-11-01.md)。
- 对中央舵机采用更强的假设仿真控制时，落地小幅转向响应有所改善，但仍未达到目标；更激进增益出现较高内部接触力。具体对照记录在 `simulation/hip_servo_gains_diagnostic.json`，参数不能直接用于实机。
- 进一步使用 SolidWorks 2025 检查保存姿态和假设髋部 ±5° 内的 11 个离散姿态，所选机身/腿架 CAD 实体未发现体积交叠；假设齿轮活动范围的 41 个离散采样点也未发现齿轮实体交叠。两项都不是完整行程或制造公差验证。
- 新增 [`diagnostic_selective_collision_trial.xml`](simulation/diagnostic_selective_collision_trial.xml)，使用更细的凸代理尝试恢复机身局部碰撞关系，仍刻意排除 79 对初始间距不足的代理对。中立和双向小幅转向短测未出现自动重置；这只验证有限测试，**碰撞覆盖不完整，不能用于训练或实机**。另几个带 `trial`、`ablation` 的模型是碰撞对照实验，其中部分会产生严重虚假接触力，不是修复版本。所有结果见 [`核验进展_2026-11-01.md`](simulation/核验进展_2026-11-01.md)。
- 后续新增 [`diagnostic_range_collision_trial.xml`](simulation/diagnostic_range_collision_trial.xml)：对假设髋部 ±5° 内 21 个姿态筛选机身局部凸代理，恢复 10 对、排除另 4 对，未覆盖对数由 79 变为 73。两种步长的中立站姿、小幅双向转向及 1 N 短时侧推测试均未发生仿真自动重置。筛选阈值和活动范围仅为假设，仍有 73 对及其他区域尚未完成碰撞核验，**不能用于训练或实机**。
- 脚底复核发现上述模型实际由脚部视觉网格与地面接触，原脚底盒子没有启用，且其初始最低点比原网格低约 0.39–0.52 mm。独立试验 [`diagnostic_aligned_foot_proxy_trial.xml`](simulation/diagnostic_aligned_foot_proxy_trial.xml) 将盒子最低点对齐网格并替换地面接触，但站姿时左右受力与原模型不同；盒子形状及接触区域未被实测验证，不作为当前推荐模型。详见 `simulation/foot_collision_alignment_audit.json`、`simulation/aligned_foot_ground_contact_audit.json`。
- 机身局部使用未合并的 387 块凸代理后，髋部 201 个假设姿态采样的最大代理重叠约 0.395 mm，比先前约 1.25 mm 更小，但仍有 95 对需排除，数量无法与旧代理直接比较覆盖比例。新增全配对和按假设 ±5° 行程裁剪的试验模型：6 秒双向小命令对照在记录精度下与全配对结果一致，实际测试的髋部最大变化约 0.266°。**裁剪模型没有运行时行程保护，不能用于训练或实机**。见 `simulation/nomerge_proxy_range_audit.json`、`simulation/pruned_nomerge_selection_verification.json` 和 `simulation/pruned_nomerge_motion_audit.json`。

### 复现本轮碰撞检查

在 `simulation/` 目录中，先安装上面的仿真和可选诊断依赖。运行 `python audit_highres_proxy_gap.py` 比较细分代理在中立姿态的间距；运行 `python build_selective_collision_trial.py` 重新生成部分碰撞试验并执行半秒对照，再运行 `python audit_selective_collision_motion.py` 做双向转向对照。生成脚本会覆盖同名实验 XML/JSON，请先备份需要保留的结果。

如需复现行程筛选，依次运行 `python audit_proxy_candidate_range.py`、`python build_range_collision_trial.py`、`python verify_range_collision_trial.py`，然后用 `python audit_selective_collision_motion.py --range` 和 `python audit_refined_tower_scenarios.py --range-trial` 复查有限动力学工况。这些命令会覆盖同名实验结果；它们不启动训练。

如需复现脚底对照，在 `simulation/` 依次运行 `python audit_foot_collision_alignment.py`、`python audit_ground_contact_geometry.py`、`python build_foot_proxy_trial.py` 和 `python audit_ground_contact_geometry.py --aligned-box`；这些脚本会覆盖同名诊断模型和数据，不接入真实硬件，也不训练。对齐最低点不代表整个支撑面正确。

机身未合并凸代理已保存在 `simulation/collision_nomerge/`。如需复现，运行 `python audit_nomerge_proxy_gap.py`，再运行 `python audit_nomerge_proxy_range.py` 和 `python audit_nomerge_proxy_range.py --pruned`，随后依次运行 `python build_nomerge_collision_trial.py`、`python verify_nomerge_collision_trial.py`、`python build_pruned_nomerge_trial.py`、`python verify_pruned_nomerge_trial.py`。`python audit_selective_collision_motion.py --pruned` 比较有限转向，`python audit_refined_tower_scenarios.py --pruned-trial` 检查站姿和短时侧推；`python benchmark_nomerge_collision.py` 做短时本机测速。这些命令会覆盖同名实验输出。要从原始网格重新生成代理，`refine_collision_nomerge.py` 还需要 CoACD，且可能耗时数分钟。CoACD 的块数上限及合并设置可能影响逼近误差，参见 [CoACD 原项目说明](https://github.com/SarahWeiii/CoACD)。

`audit_cad_contact_body_map.py` 和 `audit_cad_collision_sources.py` 仅限 Windows、已打开相应装配的 SolidWorks 2025 与安装了 `pywin32` 的 Python，且目前写死了 `C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM`；它们读取已打开装配、在临时实体副本上计算，不保存或关闭装配。`verify_cad_gear_phase.py --attach --25-tooth --hypothetical-range` 可在同一条件下复查假设齿轮行程。`refine_collision_highres.py` 额外需要 CoACD；仓库已保存本次生成的代理，不必为查看结果重新生成。

## SolidWorks 装配

1. 将 `cad/` 下的文件放在同一个文件夹，用 SolidWorks 2025 打开 `SimpleBiped2a_new.SLDASM`。如果软件提示定位引用，优先选择同目录的零件和子装配。
2. 模型中有多实体零件。轴线同心不单独证明舵盘和连杆刚接；更改齿轮角度前应同时检查腿架、四孔连接与舵机安装。
3. 当前 CAD 本身未写入试验得到的齿轮相位；相位仅存在于独立的仿真候选中。不要把一次齿面零干涉采样当成完整转动周期或制造公差验证。

## 在 Windows 查看 MuJoCo 演示

需要 Python 3.14，以及支持 OpenGL 的桌面。Windows 上请将仓库放在全英文路径，例如 `C:\Practice_project\Biped_Robot`：当前 MuJoCo 在这台电脑上无法从包含中文的完整目录路径加载模型。打开 PowerShell，进入本仓库：

```powershell
cd .\simulation
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-simulation.txt
# 仅在需要执行 STL 网格相交诊断时安装：
.\.venv\Scripts\python.exe -m pip install -r requirements-diagnostics.txt
```

然后在资源管理器中双击：

- [`查看齿轮腿部联动演示.bat`](simulation/查看齿轮腿部联动演示.bat)：只播放运动学动画，让小齿轮和与大齿轮相连的两侧腿架反向摆动；**不计算重力、力矩或碰撞，也不训练**。已验证窗口可以打开。
- [`查看站姿控制候选.bat`](simulation/查看站姿控制候选.bat)：使用实验舵机控制器和地面接触运行站姿候选；不训练。该窗口在本次整理时仅通过脚本语法检查，若无法打开，请检查依赖和终端报错。

两者均可直接关闭窗口结束。普通 MuJoCo Viewer 直接打开 XML 不会自动运行 Python 舵机控制器，默认姿态也未必是 `centered_start` 关键帧。旧实验 XML 和训练脚本均保留供追溯，不能默认当作当前推荐模型。

部分旧脚本仍引用创建者电脑上的绝对路径，如 `C:\Practice_project\biped_sim`、`C:\Practice_project\fangzhen` 或 `E:\anaconda3`；这些脚本在其他位置运行前需要先检查并修改路径，少量 SolidWorks 脚本要求 Windows、SolidWorks 2025 和 `pywin32`。脚本可能生成或覆盖它写明的输出文件，请先查看源代码。GPU/Warp 本地依赖缓存未提交，相关实验需要另行配置；不能仅凭本仓库宣称 GPU 环境已验证。

## ESP32 和硬件

用 VS Code 打开 `firmware/`，它包含 `platformio.ini`，配置 `esp32dev`、Arduino 和项目依赖。硬件实际型号、舵机零位、负载响应、电源和连接均需先核实。主程序此前通过编译，但尚未在实物上完整验证；单项测试顺序见 [`到货后调试说明.md`](firmware/到货后调试说明.md)。不要把仿真假设直接用于实物动作。

## 历史检查点：不上传压缩包

旧 Stable-Baselines3 检查点原先是 `.zip`；仓库没有提交这些 ZIP，而将其中的 `data`、`policy.pth`、`policy.optimizer.pth`、`pytorch_variables.pth`、版本及环境信息逐个保存于 `simulation/模型检查点/`。`整理记录.json` 记录每个展开目录对应的原 ZIP 路径。

如确实需要供原有 `PPO.load()` 使用，可在本地重建某个检查点，例如：

```powershell
python .\tools\恢复模型检查点.py .\simulation\模型检查点\continued_gpu_98304 --output "$env:TEMP\continued_gpu_98304.zip"
```

输出只用于本地，不应提交到仓库；脚本不会覆盖已有输出，也不会启动训练。旧模型依赖当时的软件版本及环境，不代表可靠行走策略。

不提交 `.venv`、`.pio`、下载的 GPU 依赖目录和缓存，也不提交 ZIP、7z、RAR、tar 等通用压缩包。`.3mf` 是打印工程格式，`.npz` 是 NumPy 数据，`.pth` 是模型数据，它们作为项目资料保留。后续添加或修改文件时运行 `python .\tools\生成文件索引.py` 并核对更新后的文档。
