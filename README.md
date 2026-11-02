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

**2026-11-02 左脚更新：** 用户确认 SolidWorks 中已压缩左脚外侧浮空板。只读检查当前打开的装配，两脚均各有两个可见实体；旧仿真左脚网格多出的一块约 125 × 2 × 10 mm 薄条对应旧质量清单中的额外实体。独立候选 `diagnostic_foot_suppressed_plate_trial.xml` 移除了该薄条和约 2.94 g 的估算质量；原模型保留。脚部网格参与地面碰撞，相关历史结果应在新模型上重测。

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
- 机身局部使用未合并的 387 块凸代理后，髋部 201 个假设姿态采样的最大代理重叠约 0.395 mm，比先前约 1.25 mm 更小，但仍有 95 对需排除，数量无法与旧代理直接比较覆盖比例。新增全配对和按假设 ±5° 行程裁剪的试验模型：6 秒双向小命令对照在记录精度下与全配对结果一致，实际测试的髋部最大变化约 0.266°。**裁剪模型 XML 本身没有内置运行时行程保护，不能用于训练或实机**。见 `simulation/nomerge_proxy_range_audit.json`、`simulation/pruned_nomerge_selection_verification.json` 和 `simulation/pruned_nomerge_motion_audit.json`。
- 可选的 [`guarded_pruned_servo.py`](simulation/guarded_pruned_servo.py) 为 Python 仿真诊断提供 ±4.5° 停止检查，测试 `python audit_pruned_range_guard.py`。**保护只在该控制器运行时生效**；直接打开 XML、使用其他控制器、单步跨越阈值、未建碰撞和实际硬件均不受其保护。它不能作为训练许可或实机安全控制。
- 更细代理试验 [`diagnostic_finer_collision_trial.xml`](simulation/diagnostic_finer_collision_trial.xml) 使用 822 块机身代理和两侧 315 / 348 块腿架代理，在 201 个假设姿态中最大代理重叠约 0.178 mm，仍有 89 对近距离代理被排除。与旧模型的 95 对来自不同分解，不能直接比较物理覆盖比例。6 秒小幅双向转向和有限站姿、侧推对照未自动重置；仍不用于训练或实机。测试该模型的可选控制器检查可运行 `python audit_pruned_range_guard.py --finer`。
- 另一机身部件的独立试验 [`diagnostic_chassis0_collision_trial.xml`](simulation/diagnostic_chassis0_collision_trial.xml) 将该部件分解为 544 块代理，假设髋部行程内最大代理重叠约 0.080 mm；其 820 对近距离代理仍被排除。较高增益的小幅正向转动对照中，采样最大单点内部接触力从约 59.77 N 降到 17.03 N，不能据此推断实机力或完整碰撞。该模型只用于诊断，不用于训练或实机；可选角度检查使用 `python audit_pruned_range_guard.py --chassis0`。
- 原始 STL 源网格在 21 个髋部采样姿态的 FCL 对照中，机身部件 0 的三角面与两侧腿架网格均有相交，而 SolidWorks 实体的离散检查未检测到体积重叠；这两种检查不等价。机身部件 5 的 STL 三角面没有检出相交，采样最小表面距离约 0.178 mm。见 `simulation/source_mesh_clearance_range_audit.json`；不能将 STL 表面距离当作 CAD 精确间隙或完整行程证明。
- SolidWorks 当前装配姿态的实体最短距离检查显示，机身部件 0 与两侧塔架均约 0 mm，最近点附近可找到方向相反的平面；机身部件 5 与两侧塔架均约 0.25 mm。初始姿态的 STL 平面投影重叠面积两侧分别约 2892 / 2891 mm²。这只是表面几何证据，并非实际有效承压面积，相关平面尚未形成可靠的仿真碰撞模型；见 `simulation/solidworks_closest_distance_audit.json` 和 `simulation/contact_plane_overlap_audit.json`。
- 平面三角片的独立试验 [`diagnostic_cad_plane_contact_trial.xml`](simulation/diagnostic_cad_plane_contact_trial.xml) 在 1 秒站姿中没有采样到平面接触力；加入检测余量的 [`diagnostic_planar_detection_only.xml`](simulation/diagnostic_planar_detection_only.xml) 则生成约 11200 个平面接触并出现异常数值力，**动力学无效，不可训练或实机使用**。两侧各 100 个样本的法向相对运动雅可比接近零，说明相应方向已被现有理想铰链约束；真实表面摩擦、预紧及受力尚未建模。详见 `simulation/planar_detection_only_audit.json`。
- 对髋关节假设每侧 0–0.1 N·m 的阻力上限做小幅双向灵敏度测试，结果在本组动作中变化很小；这不是实测摩擦参数。MuJoCo 的 [`frictionloss`](https://mujoco.readthedocs.io/en/stable/computation.html#friction-loss) 是不随接触法向力变化的简化约束，不能直接代表接触面的摩擦或预紧。见 `simulation/hip_frictionloss_sensitivity_audit.json`。

### 复现本轮碰撞检查

在 `simulation/` 目录中，先安装上面的仿真和可选诊断依赖。运行 `python audit_highres_proxy_gap.py` 比较细分代理在中立姿态的间距；运行 `python build_selective_collision_trial.py` 重新生成部分碰撞试验并执行半秒对照，再运行 `python audit_selective_collision_motion.py` 做双向转向对照。生成脚本会覆盖同名实验 XML/JSON，请先备份需要保留的结果。

如需复现行程筛选，依次运行 `python audit_proxy_candidate_range.py`、`python build_range_collision_trial.py`、`python verify_range_collision_trial.py`，然后用 `python audit_selective_collision_motion.py --range` 和 `python audit_refined_tower_scenarios.py --range-trial` 复查有限动力学工况。这些命令会覆盖同名实验结果；它们不启动训练。

如需复现脚底对照，在 `simulation/` 依次运行 `python audit_foot_collision_alignment.py`、`python audit_ground_contact_geometry.py`、`python build_foot_proxy_trial.py` 和 `python audit_ground_contact_geometry.py --aligned-box`；这些脚本会覆盖同名诊断模型和数据，不接入真实硬件，也不训练。对齐最低点不代表整个支撑面正确。

机身未合并凸代理已保存在 `simulation/collision_nomerge/`。如需复现，运行 `python audit_nomerge_proxy_gap.py`，再运行 `python audit_nomerge_proxy_range.py` 和 `python audit_nomerge_proxy_range.py --pruned`，随后依次运行 `python build_nomerge_collision_trial.py`、`python verify_nomerge_collision_trial.py`、`python build_pruned_nomerge_trial.py`、`python verify_pruned_nomerge_trial.py`。`python audit_selective_collision_motion.py --pruned` 比较有限转向，`python audit_refined_tower_scenarios.py --pruned-trial` 检查站姿和短时侧推；`python benchmark_nomerge_collision.py` 做短时本机测速。这些命令会覆盖同名实验输出。要从原始网格重新生成代理，`refine_collision_nomerge.py` 还需要 CoACD，且可能耗时数分钟。CoACD 的块数上限及合并设置可能影响逼近误差，参见 [CoACD 原项目说明](https://github.com/SarahWeiii/CoACD)。

复现更细候选：在 `simulation/` 依次运行 `python audit_nomerge_proxy_gap.py --legs --finer`、`python audit_finer_proxy_range.py`、`python build_finer_collision_trial.py`、`python verify_finer_collision_trial.py`；然后运行 `python audit_selective_collision_motion.py --finer` 和 `python audit_refined_tower_scenarios.py --finer-trial` 做有限动态测试。脚本覆盖对应实验文件，不启动训练；`simulation/collision_finer/` 和 `simulation/collision_nomerge/L0`、`L1` 保存本次生成的代理，重新分解需要 CoACD。

复现另一机身部件的试验：在 `simulation/` 先运行 `python audit_chassis0_proxy_gap.py --finer`、`python audit_finer_proxy_range.py --chassis0`，再运行 `python build_chassis0_collision_trial.py` 和 `python verify_chassis0_collision_trial.py`。动态对照可运行 `python audit_hip_integral_trial.py --chassis0 --positive-ki10`、`python audit_selective_collision_motion.py --chassis0`、`python audit_refined_tower_scenarios.py --chassis0-trial`。这些命令覆盖相应实验输出，不训练；范围裁剪和缺失碰撞对的限制仍然适用。

如需比较原始 STL 表面，可在安装可选诊断依赖后运行 `python audit_source_mesh_clearance_range.py`；会覆盖对应 JSON，不能替代 SolidWorks 实体与实物的间隙核验。

复现当前装配接触测量：在 SolidWorks 2025 中打开原路径 `C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM`，使用装有 `pywin32` 的 Python 在 `simulation/` 依次运行 `audit_solidworks_closest_distance.py` 和 `audit_solidworks_contact_faces.py`；脚本只读测量，不保存或关闭装配。再用带 MuJoCo、Trimesh 等诊断依赖的 Python 依次运行 `audit_cad_contact_mapping.py`、`audit_cad_contact_planes.py`、`audit_contact_plane_overlap.py` 对照仿真源网格。脚本会覆盖同名 JSON；不同姿态不能直接比较。

独立平面试验可运行 `python build_planar_contact_trial.py`、`python audit_planar_contact_trial.py` 和 `python audit_planar_contact_margin.py` 复现；会覆盖同名试验模型、网格与数据。后一个检测只作初始状态诊断，已知产生异常数值力，不能作为运行或训练模型。

在 `simulation/` 运行 `python audit_hip_frictionloss.py` 可重新计算髋部阻力的八组假设对照（约需数分钟）；`python audit_hip_frictionloss.py --smoke` 只运行一组并检查两个阻力约束。命令会覆盖相应 JSON，只在内存中修改参数，不启动训练。

`audit_cad_contact_body_map.py` 和 `audit_cad_collision_sources.py` 仅限 Windows、已打开相应装配的 SolidWorks 2025 与安装了 `pywin32` 的 Python，且目前写死了 `C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM`；它们读取已打开装配、在临时实体副本上计算，不保存或关闭装配。`verify_cad_gear_phase.py --attach --25-tooth --hypothetical-range` 可在同一条件下复查假设齿轮行程。`refine_collision_highres.py` 额外需要 CoACD；仓库已保存本次生成的代理，不必为查看结果重新生成。

### 站姿训练环境与轴向接触诊断

`sector_standing_env.py` 是基于分段候选的实验性 Gymnasium 站姿环境，不包含训练启动程序。五个动作映射到有限的假设舵机 PWM 偏移；每个物理子步检查非有限状态、仿真重置、髋轴相对初始姿态超过 ±4.5° 和接触容量，触发时结束回合。回合信息记录采样内部接触力及齿轮约束误差；力每 10 个物理子步采样一次，可能漏掉尖峰，未设置实测安全阈值。保护只在该环境的 Python 步进中生效，可能在一个物理子步内越过边界，不能保护 XML 直接运行或实物。可选参数 `max_sampled_self_contact_n` 可设置仿真内部接触力采样截断，例如 `SectorStandingEnv(max_sampled_self_contact_n=20.0)`；默认不启用。20 N 只是用于测试的假设数值，不是实体安全限值。`python audit_sector_sampled_force_guard.py` 验证双向高力回合会结束和较高阈值对照；实际采样可在超过阈值之后才检测到，间隔内尖峰可能漏检。`python audit_sector_mixed_action_smoke.py` 验证三个低幅度组合动作轨迹，均仅 2 秒。观测假设可获得机身重力方向、角速度和控制命令，奖励只针对站姿，尚不是走路奖励；尚未验证真实传感器延迟或舵机反馈。

`python audit_sector_standing_env.py` 检查确定性复位、无效动作、结束后再步进、髋轴越界和非有限状态；`python audit_sector_standing_env_sustained.py` 检查两个持续中央动作，`python audit_sector_standing_env_single_axis.py` 单独检查五轴双向动作。它们不启动训练，覆盖同名 JSON。Gymnasium 和 Stable-Baselines3 接口检查通过；Gymnasium 对观测空间使用无穷界发出提示。十个单轴双向 2 秒测试均正常结束，1 秒随机动作也已测试；组合动作和更长场景仍需检验。`audit_sector_training_readiness.py` 汇总当前保守判断，输出 `sector_training_readiness_audit.json`；它不是训练启动器，也不是实物安全认证。

`audit_cad_peak_force_pair.py` 在打开的 SolidWorks 2025 原装配中，以临时实体副本检查高接触力涉及的机身侧板与左腿架；11 个假设角度未发现体积干涉。`audit_sector_peak_contact_direction.py` 在一组 -20 µs 指令下记录该代理采样峰值约 35 N，法向与理想髋轴接近平行；`audit_peak_pair_axial_bounds.py` 检查两代理轴向边界及刚性父子关系。该数值力是仿真中的采样值，不等于实物力。

`diagnostic_sector_axial_contact_trial.xml` 仅在独立候选中移除此对显式接触。`audit_sector_motion_stress.py --axial` 的双向短测显示采样内部接触峰值消失，但控制响应发生明显变化、一个方向出现较大超调；`audit_axial_contact_gear_response.py` 还记录齿轮约束误差。删除该对同时会删除接触摩擦，理想铰链不模拟真实预紧和摩擦，因此该候选未被选作训练模型。`audit_sector_axial_damped_motion.py` 和 `audit_sector_axial_low_integral_motion.py` 是不同假设控制参数的对照，不代表实测或推荐参数。

`build_sector_gear_stiff_trial.py` 和 `audit_sector_gear_stiff_motion.py` 构建并测试仅修改两条齿轮等式数值刚度的独立候选；`audit_sector_gear_stiff_halfstep.py` 做较小时间步长对照。较硬候选负向 6 秒测试的最大齿轮约束误差约 0.046°，原候选约 0.106°；步长减半后结果接近，但高接触力仍然存在且正向跟踪误差没有改善。更硬的等式约束不等同真实齿轮刚度，也不会自动解决高接触力。相关脚本覆盖同名 XML/JSON；实际舵机参数、碰撞覆盖及训练策略仍未验证，不能用于实机。

### CAD 加密碰撞检查和简化阻力试验

`cad_collision_source_range_dense.json` 保存 SolidWorks 2025 对两个机身实体与两侧腿架在假设髋部 ±5° 的 101 个离散角度检查（共 404 次），无体积重叠；`verify_cad_collision_dense_coverage.py` 可验证角度覆盖和结果。原始测量由 `audit_cad_collision_sources.py --range --dense` 在打开的原装配上使用临时实体副本完成，不保存或关闭装配。离散零重叠不是连续运动或所有零件碰撞证明。

`audit_sector_axial_joint_friction_motion.py` 在内存中给去掉疑似重复接触的独立候选添加每侧假设 0.1 N·m 的关节阻力，对比 ±20 µs 指令；`sector_axial_joint_friction_motion_audit.json` 保存结果。负向末端仍超调约 1.29°，不能据此替换原接触或认定真实摩擦；无训练和硬件操作。

### 髋部孔边凸代理分段候选

对上轮 58 个距 CAD 面超过 0.1 mm 的机身代理见证点，用对应封闭 STL 做近似点内外检查：58 个都在 STL 外，其中 57 个靠近孔边。这支持这些代理有外扩的可能，但 STL 检查不能代替准确 CAD 实体布尔判断。SolidWorks 只读测量确认相应机身实体有两个半径 20 mm、轴向范围 136–146 mm 的孔。

独立候选 `diagnostic_sector_collision_trial.xml` 将所涉及的 18 个凸代理按最大 3° 扇区裁剪，得到 154 个封闭凸块。相对于理想圆孔，分段裁剪多移除的最大径向范围数学上约 0.0069 mm；该界只对所用圆柱假设成立，不代表 CAD 全形状精度。154 个导出网格均通过圆孔半径与闭合检查。原模型和其他区域不变；154 块的接缝可能产生重复接触，必须继续核验。

对先前抽查的 57 对碰撞，假设髋轴 ±5° 范围的 201 点采样及相邻采样间运动上界表明其中 56 对有正间距下界；候选只恢复这 56 对，另 1 对继续排除。原 120 对已启用的代理组合被映射到分段块，其他碰撞关系保留。五组 4 秒对照（中立、步长减半、双向短推及小幅指令）完成，站姿、倾角和采样内部接触力与原候选在记录精度下相同，未检出接触容量异常。该机身部件仍有约 734 对近距离原始代理组合未恢复，其他部件也有排除。XML 无行程保护，旧 `GuardedPrunedServo` 不支持该候选；仅下方情景脚本逐步检查假设 ±4.5° 边界。模型不用于正式训练或实物。

复现（先按前文生成 CAD 分类结果，并安装仿真依赖和 `python -m pip install pywin32 rtree`）：打开 SolidWorks 2025 中已保存的原装配，在 `simulation/` 依次运行：

```powershell
python audit_cad_chassis0_holes.py
python audit_chassis_witness_stl_containment.py
python audit_tangent_clip_feasibility.py
python audit_sector_clip_prototype.py
python verify_sector_mesh_holes.py
python build_sector_gap_only.py
python audit_sector_gap_only_witness.py
python audit_sector_gap_only_sweep.py
python build_sector_collision_trial.py
python verify_sector_collision_selection.py
python audit_refined_tower_scenarios.py --sector-collision
```

命令覆盖相应的实验 STL、XML 和 JSON。`collision_sector_trial/chassis0/` 保存分段凸块；各 `*_audit.json` 记录 CAD 圆孔、STL 包含关系、裁剪几何及有限采样距离；`sector_collision_trial_build.json` 记录碰撞组合，`sector_collision_selection_verification.json` 校验组合与参数，`sector_collision_scenarios_audit.json` 保存站姿、轻推和小指令对照。采样不覆盖实际制造偏差、所有关节动作或全部接触。

### 剩余近距离代理的 CAD 面分类

在机身部件仍被排除的 790 对代理中，选出初始姿态代理间距在 0.0001–0.1 mm、既有行程采样最小间距不超过 0.05 mm 的 141 对，与当前打开的 SolidWorks 2025 装配做最近面比对；其余 649 对不在本次样本内。64 对的两侧代理最近点距离各自最近 CAD 平面均小于 0.01 mm，且两面法向相反；在假设髋角 -5°、0°、+5° 的局部速度雅可比中，模型里这些对沿平面法向的相对运动分量为零（记录精度内）。这说明该方向已经由理想铰链约束，是继续排除重复代理接触的模型依据，不说明实物预紧、接触力、摩擦、全部角度或全部排除对正确。另有 58 对的机身代理最近点到最近 CAD 面超过 0.1 mm，提示局部代理近似与 CAD 可能存在偏差；这不是两实体的实际间隙或干涉量。

在 Windows 启动 SolidWorks 2025 并打开 `C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM`；在装有 MuJoCo、NumPy 和 `pywin32` 的 Python 环境进入 `simulation/`，依次运行：

```powershell
python -m pip install pywin32
python audit_remaining_collision_cad_witness.py
python audit_coplanar_witness_hinge_jacobian.py
```

两脚本只读已打开装配和临时实体副本，不保存或关闭装配，覆盖对应 JSON。`remaining_collision_cad_witness_audit.json` 记录所选代理及最近 CAD 面；`coplanar_witness_hinge_jacobian_audit.json` 记录 64 对在三个角度的法向雅可比。CAD 与仿真坐标对应仍是近似；距离是代理见证点到单个 CAD 面，并非实体间准确间距。碰撞仍不完整，候选不能用于正式训练或实机。

### 机身与腿架近距离碰撞对恢复试验

`diagnostic_restored_near_pairs_trial.xml` 在脚部拆分候选上恢复 30 对机身—腿架凸代理碰撞。它们原先因采样最小间距不超过 0.1 mm 被排除；对假设髋部 ±5° 做 2001 个离散采样，并按各刚体代理到转轴的最大半径计算相邻采样点之间可能的最大移动量后，这 30 对的估算最小间隙下界均高于 0.049 mm。这是理想刚体代理和数学采样下界，并非 CAD 实际间隙或生产公差保证；在此假设行程内这些对本来就不应接触，恢复它们不等于解决了真实接触或显著增加了有效碰撞覆盖。

该机身部件仍有 790 对近距离代理未启用，其他部件还有额外排除；不适合正式训练或实机。XML 本身没有范围保护，不能直接使用旧 `GuardedPrunedServo`（它尚未适配此模型）；下列情景脚本为本候选逐步检查 ±4.5° 的假设边界及接触容量，检查只在该脚本运行时有效。

在 `simulation/` 安装仿真和诊断依赖后运行：

```powershell
python audit_near_collision_gap_refinement.py
python build_restored_near_pairs_trial.py
python audit_restored_near_pairs_difference.py
python audit_refined_tower_scenarios.py --restored-near
```

命令会覆盖对应 XML 和 JSON。`near_collision_gap_refinement_audit.json` 保存每对采样最小距离及运动界，`restored_near_pairs_build.json` 记录恢复数量，`restored_near_pairs_difference_audit.json` 验证其余配置未变，`restored_near_pairs_scenarios_audit.json` 记录站姿、轻推和小幅指令测试。中立、步长减半、双向 1 N 短推和小幅指令的 4 秒对照均完成，结果与原候选接近。`audit_remaining_collision_axis.py` 和 `remaining_collision_axis_classification.json` 给剩余 790 对记录采样最小间距时最近点的柱轴半径及轴向位置，供后续对照 CAD；最近点和半径不能证明表面身份，重叠时最近点尤其不可靠。该试验不解决被排除的真实近接触问题。

### 脚部拆分碰撞候选

`diagnostic_separate_sole_contact_trial.xml` 基于左脚薄板压缩候选，把每只脚的脚底和脚体拆成独立地面碰撞几何：脚底和脚体分别可与地面接触，视觉、质量惯量、腿部自碰撞及控制不变。网格来自现有 STL 中的两个封闭连通体，各自仍采用凸包，其他部位的碰撞排除仍然存在，不用于正式训练或实机。

`audit_foot_convex_surface.py` 在初始及绕世界 X/Y 单独倾斜 ±5°、±15° 的几何姿态取网格样本；离最低点 1 mm 内采样点没有发现超过 0.5 mm 的原始表面与整体凸包高度差，不能替代完整运动和实际接触验证。`audit_separate_sole_ground_distance.py` 在同样的单轴姿态比较新旧地面最小距离，最大差小于 0.000001 mm；最小距离不代表接触力、摩擦和完整接触区域相同。

在 `simulation/` 安装仿真和诊断依赖后可运行：

```powershell
python audit_foot_convex_surface.py
python build_separate_sole_contact_trial.py
python audit_separate_sole_model_difference.py
python audit_separate_sole_ground_distance.py
python audit_pruned_range_guard.py --separate-sole
python audit_refined_tower_scenarios.py --separate-sole
```

独立候选完成 4 秒中立、步长减半、双向 1 N 短时推力及小幅指令对照，结果与原脚部候选接近；这些只是有限场景。

生成脚本覆盖相应 STL、XML 和 JSON。`meshes_foot_soles/` 中四个文件是拆出的左右脚底和脚体；`separate_sole_contact_build.json` 记录构建结果；`separate_sole_model_difference_audit.json` 核验预期 XML 差异；`separate_sole_ground_distance_audit.json` 记录平面距离；`separate_sole_range_guard_audit.json` 记录 Python 控制器限位验证；`separate_sole_contact_scenarios_audit.json` 保存站姿与短时侧推结果。范围保护仅在使用 `GuardedPrunedServo` 时生效，普通 XML、其他控制器、单步越界和真实设备不受保护。

### 脚部网格和中央舵机响应复核

运行 `python audit_foot_visual_components.py` 可以输出两侧脚部网格的连通分量及初始世界坐标包围盒，结果为 `foot_visual_components_audit.json`。旧模型一侧有一个 628 面、约 125 × 2 × 10 mm 的额外薄条，初始离地约 15 mm；独立网格检查本身不能证明实体身份；已结合用户说明、当前 CAD 可见实体及旧质量清单比对。

运行 `python audit_hip_ground_load.py` 可对 ±10 微秒阶跃比较落地和抬起无重力的 8 秒响应，覆盖 `hip_ground_load_diagnostic.json`。落地时正负指令末端误差分别约 -0.094° 和 -0.018°，说明较短测试的误差会随时间变化；两个无重力测试的末端误差约 ±0.285°，不能仅归因于地面负载，因为比较同时改变了重力和位置，且碰撞未完整。此诊断不启动训练，结果受旧脚部网格、假设舵机参数和接触影响。


左脚修正候选的 `build_foot_suppressed_plate_trial.py` 在旧网格中按连通分量识别并移除对应薄条，同时从左脚刚体的原质量及包围盒惯量估算中扣除该实体；运行会覆盖新 STL、XML 与 `foot_suppressed_plate_trial_build.json`，保留源模型。`audit_cad_foot_visible_bodies.py` 使用 SolidWorks 2025 和 pywin32 对当前打开的装配进行只读复核，输出 `cad_foot_visible_bodies_audit.json`。`python audit_refined_tower_scenarios.py --foot-suppressed-plate` 可重测站姿和轻推；`view_foot_suppressed_plate.py` 或双击 `查看左脚修正站姿.bat` 可查看该候选。五组 4 秒对照（中立、步长减半、双向短时轻推和小幅指令）均完成，数值与旧模型接近；不代表完整接触已验证。旧网格而非重新导出的 CAD 网格构成其他部分，接地网格可能被凸化，质量、惯量和碰撞仍为估算，不用于训练或实机。

### 柱面支承和理想润滑诊断

用户确认柱子、大齿轮与腿架刚性连接，柱面相对机身转动并计划润滑。SolidWorks 2025 名义孔半径为 20 mm，柱半径为 19.75 mm，名义径向间隙为 0.25 mm；两侧 CAD 轴线与仿真髋轴对齐。理想铰链没有模拟间隙、轴承径向碰撞或真实润滑摩擦。

`diagnostic_smooth_hip_trial.xml` 是独立试验模型：两侧髋关节阻尼和摩擦损失取零，其他碰撞及舵机假设保持不变。这是理想化假设，不能说明实物无摩擦。±10 微秒小幅脉宽测试中，新旧模型 4 秒末的轴角差小于 0.00002°；现有碰撞排除和传动、舵机参数不确定性仍然存在，不应用于训练或实机控制。

4 秒无虚拟辅助的中立站姿测试中，假设控制器作用下机身质心高度下降约 0.887 mm、最大倾角约 1.20°；步长减半结果接近。两侧 1 N、0.2 秒短时推力测试最大倾角分别约 2.09° 和 2.53°，不能据此保证真实站立。`view_smooth_hip_standing.py` 或双击 `查看柱面支承站姿.bat` 可查看实时站姿，需先按下文安装仿真依赖；关闭窗口即停止。该查看器采用带髋轴范围检查的控制器，参数与情景测试的第五轴略有不同。

在 `simulation/` 使用安装 MuJoCo 和 NumPy 的 Python 依次执行：

```powershell
python build_smooth_hip_trial.py
python audit_smooth_model_difference.py
python audit_hip_cylinder_alignment.py
python audit_gear_leg_rigidity.py --smooth-hip
python audit_smooth_hip_trial.py
python audit_refined_tower_scenarios.py --smooth-hip
```

这些命令会覆盖对应 XML 或 JSON 诊断结果；最后两项运行数分钟。要重新测量柱孔，需先在 SolidWorks 2025 打开原装配，再用带 `pywin32` 的 Python 运行 `audit_cad_hip_cylinders.py`。脚本只读取装配并使用临时实体副本，不保存或关闭原装配；`cad_hip_cylinders_audit.json` 和 `hip_cylinder_alignment_audit.json` 保存测量和轴线对照结果。`smooth_hip_model_difference_audit.json` 记录 XML 差异；`smooth_hip_gear_leg_rigidity_audit.json` 记录腿架与小齿轮的运动学层级；`smooth_hip_trial_audit.json` 保存小幅指令对照，`smooth_hip_scenarios_audit.json` 保存站姿和推力情景。具体限制及每个文件用途见核验进展和文件索引。

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


## 实验：隐式阻尼与 GPU 并行站姿环境

本节是仿真诊断原型，不是行走策略，也没有启动训练。`diagnostic_sector_implicit_damping_trial.xml` 与 `diagnostic_sector_implicit_stiff_trial.xml` 是从候选模型生成的实验版本：移除了一个轴向法向接触，并假设每侧髋轴摩擦 0.1 N·m、阻尼 0.301 N·m·s/rad；这些量尚未实测，不能视为真实舵机参数。原显式控制微分项产生约 14 rad/s 的启动速度峰值，改成被动物理阻尼后该测试约为 0.9 rad/s 以下，但控制误差和步长敏感性仍未解决。减小步长或限制 PWM 变化速率的实验并未证明结果收敛。

`sector_standing_env.py` 支持实验模型、假设的摩擦范围、齿轮误差与采样接触力阈值、PWM 变化速率限制；这些阈值是诊断值，不是安全认证。`sector_warp_standing_vec_env.py` 使用 CUDA 上的 MuJoCo Warp 并行物理，当前 GPU 版本使用固定假设摩擦；`sector_warp_guard.py` 和 `sector_warp_servo_kernels.py` 分别负责状态与接触检查、实验舵机控制。默认异常时终止整个批次，`on_guard="terminate"` 可试验单环境重置，其隔离性目前只做了短时验证。接触力每 2.5 ms 采样一次，可能漏掉尖峰；溢出或异常需要中止。

本机 16 个环境、4 秒混合小动作测试约 131 环境步/秒，数字包含该次测试的同步开销，不保证其他电脑相同。GPU 用于物理计算，本机当前 PyTorch 为 CPU 版；每个物理子步的主机同步会限制利用率。4 环境、8 秒的部分 CPU/GPU 轨迹对照和 1 秒重置隔离检查不等于完整一致性验证。SB3 只检查了创建、预测和单步接口，没有调用学习。旧训练准备检查不自动覆盖此 GPU 原型。

使用前在 Python 环境中安装项目所需依赖和匹配版本的 Warp，例如 `python -m pip install warp-lang==1.18.0 mujoco-warp==3.17.0`，并确认 NVIDIA 驱动与 CUDA 设备可用；依赖缓存不纳入仓库。在 `simulation` 目录运行：

```powershell
python .\audit_sector_warp_vec_smoke.py
python .\audit_sector_warp_per_world_guard.py
python .\audit_sector_warp_vec_16_4s.py
```

这些脚本运行短时仿真并覆盖对应的 JSON 结果，不会训练；其他 `audit_sector_*`、`build_sector_*` 文件的用途及运行方式见文件清单。碰撞代理、齿轮约束的数值收敛、实际舵机与负载参数和行走奖励仍需核实，不能据此把仿真动作下发到实体机器人。


### 初始倾斜与地面接触检查

零动作 16 环境、10 秒检查未见倒地或保护触发；八个单轴 ±1°、±2° 初始倾斜在修正脚底初始穿地后完成 4 秒。直接绕自由关节旋转会使脚底深入地面，最多约 3.25 mm；碰撞冲击曾使一对机身/塔架代理出现约 107 N 的采样力，因此初始化必须检查地面距离，不能靠关闭碰撞解决。

`sector_reset_pose.py` 提供初始倾斜和地面抬升计算，`SectorWarpStandingVecEnv(reset_tilt_deg=2.0)` 可选对横滚和俯仰分别随机采样 ±2°，默认仍关闭随机倾斜；`env.seed(314)` 后 `env.reset()` 可重现采样。随机重置时只更改选中环境的初始位置，开始前以碰撞代理的地面接触距离修正高度。坐标和倾角是仿真假设，不能作为真实倾倒安全范围。环境还记录每帧采样峰值的接触代理名称。

`audit_sector_initial_tilt_geometry.py` 测初始穿地；`audit_sector_warp_neutral_long.py`、`audit_sector_warp_tilt_response.py`、`audit_sector_random_tilt_reset.py` 和 `audit_sector_random_tilt_episode.py` 检查零动作与随机重置，后者可指定 1 或 2 度：`python audit_sector_random_tilt_episode.py 1`。一次较早候选测试中四个随机环境有一个超过 20 N 保护阈值，后续加入碰撞对记录后的两次同种子检查均未触发；它们使用不同版本的诊断代码，不能当成严格的可重复性结论。即使通过短测试，也没有验证复杂动作、步长收敛、未覆盖的碰撞和真实舵机参数。


### 站姿奖励候选

`sector_standing_reward.py` 是仅供仿真的站姿奖励：直立得分，机身下降、角速度、动作幅度及动作变化扣分，终止额外扣分。权重是待验证的实验假设，奖励使用仿真器中的高度，仅说明训练时可以取得该信号；它并不意味着实物能直接测到机身高度。奖励不包含向前速度、迈步或左右脚交替，因此**不能单靠它训练行走**。

CPU 和 GPU 环境均可使用 `reward_mode="shaped"` 选择候选，默认仍是 `legacy`，以保留旧试验结果的含义。GPU 可另用 `reset_tilt_deg=1.0`、`on_guard="terminate"` 进行诊断。`audit_sector_shaped_reward.py` 核对静态奖励偏好及 CPU/GPU 短序列，`audit_sector_shaped_policy_interface.py` 只验证未训练 PPO 模型的一次预测与环境步进，`audit_sector_shaped_random_actions.py` 测小幅随机动作；它们不调用学习，也不保存策略。短时间对照无法证明奖励避免投机、能够收敛或可用于实物。


### 零动作时间步长对照

`audit_sector_neutral_timestep.py` 将相同的隐式阻尼候选和假设舵机控制器分别以 0.25、0.125、0.0625 ms 的物理步长运行 1 秒；不训练。该次零动作检查中，相邻步长的一秒末机身自由关节位置相差约 0.0088 mm 和 0.0042 mm，最终舵机力矩最大差分别约 0.000092 N·m 和 0.000015 N·m。比较的是数值一致性，不是位置精度或实物验证。此前髋轴舵机阶跃和缓慢指令的时间步长对照仍有较大差异，因此尚不能据此认为动态响应已经收敛。


`audit_sector_timestep_friction.py` 再对比髋轴摩擦设为 0 和 0.05 N·m、步长设为 0.25 和 0.125 ms 的四种假设：在 0.5 秒施加 -20 μs 指令，到 2 秒时四种情况下目标与舵机轴位置仍差约 -2°。这是控制器和模型在该时刻的跟踪偏差，不应直接解释为实物误差；也不能仅凭这个测试断定偏差来源。仿真关节摩擦、齿轮约束、控制器和舵机负载响应仍需更完整的动作对照。


`audit_sector_servo_gain_sweep.py` 对假设的中央舵机比例与积分增益做三组 3 秒阶跃检查：同一 -20 μs 指令在仿真结束时，三组轴角度约为 -3.83°、-1.03°、-1.44°，目标约为 -2.7°。实验显示假设控制器会显著影响结果；这三组并非实测增益，且未做完整碰撞、双向动作或多时间步长验证，不能选择一组作为真实舵机配置。
