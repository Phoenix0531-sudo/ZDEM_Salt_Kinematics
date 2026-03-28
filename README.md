# ZDEM 离散元盐构造数值模拟：运动学提取与可视化管线

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![ZDEM](https://img.shields.io/badge/ZDEM-Simulation-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **TL;DR**: 一套针对离散元 (ZDEM) 盐流变模拟的高并发分析管线，将海量原始空间物理帧数据零损耗特征提取，并持久化为运动学几何特性曲线，支持高精度出图对比。

## 1. 概述
本项目提供了一套专为 ZDEM (Z-Discrete Element Method) 盐岩构造模拟结果定制的 Python 数据分析与可视化管线。其核心目的在于通过离散颗粒几何信息的提取与时间维度的特征追踪，实现从原始 `.dat` 文件到运动学指标演化曲线的高效转化。

为了优化离散元海量数据的 I/O 开销，本管线采用了**数据提取**与**图形渲染**相解耦的架构。此架构彻底避免了因前端绘图参数微调而导致的后台数据重复解析问题。

## 2. 环境依赖 (Dependencies)
在运行脚本之前，请确保 Python 环境中已安装以下库：
```bash
pip install numpy pandas scipy matplotlib tqdm
```

## 3. 核心文件与系统架构 (Architecture & Modules)
此系统由三个协同工作的 Python 脚本组成：

### 📌 `01_data_extractor.py` (后台并发数据提取器)
该脚本作为数据处理引擎，负责纯后台批处理任务：
- **算法加速**：基于 `concurrent.futures.ProcessPoolExecutor` 实现多进程池，对实验组采用“组间串行、组内数据帧并行”的计算策略。
- **形态追踪**：使用 `binned_statistic` 提取表层离散颗粒物理包络线（Envelope），引入 Savitzky-Golay 滤波拟合去噪，并结合 `find_peaks` 搜寻盐丘隆起的基准特征点。
- **真实刺穿点精确捕捉**：预防强行均匀下采样导致的关键帧丢失（定位 Extruded Area > 0 的初次发生时刻），并精准索引进最终采样序列。
- **数据持久化**：脚本中没有任何可视化代码，直接将物理帧解析结果持久化输出为时间特征序列表 `kinematics_data.csv` 与轮廓数据缓存字典 `profiles_cache.pkl`。

### 📌 `02_plot_single_diagnostic.py` (单组剖面诊断图渲染器)
用于快速审视单个实验组的形态学剖面及渲染质量验证：
- **切片诊断**：解析 `profiles_cache.pkl`，使用 `fill_between` 实心填充盐体剖面掩膜，并精准标记等效基底与刺穿突破点。
- **强制锁定全局统一坐标系**：锁定几何空间尺度域（例如 X: [0, 80,000]，Y: [0, 30,000]），坚决阻断由于不同演化阶段数值范围伸缩而引起的视觉畸变。
- **配置热渲染**：只需修改代码顶部的 `TARGET_GROUP_INDEX` 变量并运行，即可瞬间刷新目标实验组的诊断图 `Salt_Profiles_Diagnostic_Grid.png`。

### 📌 `03_plot_multi_comparison.py` (多组联合演化对比渲染器)
提取不同运动学边界条件（如不同基底缩短速率 $v$）主导的多相同物理组的演化指标，并投射到统一的终极对比图中：
- **分段样式平滑闭合**：依据 CSV 中的截断节点严格映射曲线特性。地下运移阶段（Pre-extrusion）使用带标记位的实线表示，地表流溢阶段（Post-extrusion）切换至无符号虚线，保证拓扑节点完全平滑吻合。
- **剥离外源修饰机制**：遵循 Tufte 极简绘图原则，移除右上冗余边框（Spines），外挂标签化体系（仅保留底部和右侧），并全局使用统一代理图例（Proxy Artist assigned to Dashed lines）。

## 4. 目录结构 (Directory Tree)
执行整套管线前后，工作空间的标准目标目录树如下所示：

```text
Working_Directory/
│
├── 01_data_extractor.py              # 数据提取脚本
├── 02_plot_single_diagnostic.py      # 单组诊断绘图脚本
├── 03_plot_multi_comparison.py       # 多组对比绘图脚本
├── README.md                         # 本说明文档
│
├── 最终出图目录/                     # (对应 FINAL_OUTPUT_DIR) 综合对比图汇总区
│   ├── Multi_Kinematic_Evolution_Plot.png
│   └── Multi_Kinematic_Evolution_Plot.pdf
│
├── 1.0/                              # 实验组 1 (例如 v=1.0) 根目录
│   ├── data/                         # 原始 ZDEM *.dat 数据帧直存中心
│   │   ├── ...
│   ├── kinematics_data.csv           # (运行 01 后生成) 运动学数据记录
│   ├── profiles_cache.pkl            # (运行 01 后生成) 轮廓数据缓存
│   ├── Kinematic_Evolution_Diagnostic.png  # (运行 02 后生成) 单组演化图
│   └── Salt_Profiles_Diagnostic_Grid.png   # (运行 02 后生成) 多宫格剖面诊断图
│
├── 2.0/                              # 实验组 2
│   ├── data/
│   ...
│
└── 3.0/                              # 实验组 3
    ├── data/
    ...
```

## 5. 标准操作规程 (SOP)

**🔥 致命警告：步骤 1：全系路径配置同步 🔥**
由于数据提取与图形渲染是**彻骨解耦**的，**请务必同时打开 `01_data_extractor.py`、`02_plot_single_diagnostic.py` 和 `03_plot_multi_comparison.py` 三个脚本**，确保它们头部的以下两项配置**一字不差完全一致**！
- `FINAL_OUTPUT_DIR`：最终联立对比出图的存放目录。
- `EXPERIMENT_GROUPS`：数据字典组别配置（包含所有的 `base_dir` 全路径指引）。

**步骤 2：解析原始物理帧**
在终端（或 IDE）中执行提取命令。由于数据规模庞大，本阶段为重 IO 瓶颈，请耐心监控终端中 `tqdm` 进度条直至 100% 满铺。
```bash
python 01_data_extractor.py
```
*(成功标志：所有物理实验组的 `base_dir` 目录下产出 `kinematics_data.csv` 与 `profiles_cache.pkl` 实体文件。)*

**步骤 3：单组深度校验诊断（可选步骤）**
如果您需要单独提取并审视各组的盐体形态学演化细节及寻峰对齐精度，请直接运行诊断渲染器：
```bash
python 02_plot_single_diagnostic.py```
(该脚本将自动遍历所有实验组，在各自的 base_dir 目录下极速生成独立的 Kinematic_Evolution_Diagnostic.png (单组演化图) 与 Salt_Profiles_Diagnostic_Grid.png (多宫格剖面校验图)，供您直观核对红色包络与粉色充填物的拟合精度。)

**步骤 4：多重演化交叉对比图绘制**
当所有数据确认无误后，直接触发终局代码：
```bash
python 03_plot_multi_comparison.py
```
前往 `FINAL_OUTPUT_DIR` 目录中打包您的顶级出版物插图素材（提供高清无损重构图：`.png`、`.pdf`）。

---
**免责声明**
- 原始数据严格遵循只读模式读取，该管线脚本绝对不会覆写、移位或剔除源文件。
- 形态提取算法针对特定几何阈值调优；当您选择改变底层骨干模型或调整离散颗粒初始半径时，请必须同步重置提取段首部的 `PARTICLE_RADIUS` 等相关常量指标。
