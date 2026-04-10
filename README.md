# ZDEM Salt Kinematics Analysis System

## 1. 项目简介
本系统是专为 **ZDEM (Zhong-Discrete Element Method)** 离散元模拟数据设计的运动学分析工具。
主要用于自动化提取盐体（Salt Body）在构造缩短过程中的几何形态演化指标，包括宽度（Width）、起伏（Relief）以及宽高比（Aspect Ratio）。

### 核心功能
- **自动化提取**: 批量解析 `.dat` 原始粒子数据，计算运动学关键点。
- **专家级质控**: 提供交互式 UI 界面，支持手动校准自动算法误判的基点。
- **学术级绘图**: 自动生成符合出版标准的演化曲线图与剖面诊断矩阵。
- **稳健性架构**: 采用并发处理与路径自动管理，支持大规模实验组比对。

---

## 2. 算法原理 (Academic Core)
1. **表面包络线提取**: 采用 `binned_statistic` 对粒子云进行分箱采样，提取上包络线。
2. **特征点定位**:
   - **主峰 (Top)**: 基于 `find_peaks` 算法结合 Prominence 过滤。
   - **基点 (Base)**: 结合局部坡度阈值 (`FLANK_SLOPE_THRESHOLD`) 与深度限制，实现复杂地貌下的边界吸附。
3. **数据平滑**: 引入自适应窗口的 `Savitzky-Golay` 滤波器，处理离散跳变噪声。

---

## 3. 快速开始 (Usage)

### 3.1 环境准备
1. 确保安装 Python 3.9+。
2. 安装依赖：`pip install numpy pandas matplotlib scipy tqdm python-dotenv`。
3. 配置 `.env` 文件（参考 `.env.example`）。

### 3.2 标准工作流
1. **[Step 1] 数据提取**:
   ```bash
   python 01_data_extractor.py
   ```
   *解析原始 .dat 文件，生成中间缓存。*

2. **[Step 2] 人工修正 (QA/QC)**:
   ```bash
   python 01b_manual_corrector.py
   ```
   *交互式检查特征点，点击图中位置可重定位基点，快捷键 [Enter] 保存。*

3. **[Step 3] 诊断渲染**:
   ```bash
   python 02_plot_single_diagnostic.py
   ```
   *生成各实验组的详细诊断图谱。*

4. **[Step 4] 跨组比对**:
   ```bash
   python 03_plot_multi_comparison.py
   ```
   *生成所有实验组的联合演化比对图。*

---

## 4. 配置指南 (Environment Variables)
| 变量名 | 描述 | 示例 |
| :--- | :--- | :--- |
| `FINAL_OUTPUT_DIR` | 数据存放的总根目录 | `./outputs` |
| `EXPERIMENT_GROUPS` | 实验组配置 (JSON 格式) | `[{"subdir": "G1", "label": "Control", ...}]` |
| `NUM_BINS` | 剖面采样分辨率 | `400` |
| `FLANK_SLOPE_THRESHOLD` | 基点识别坡度阈值 | `0.15` |

---

## 5. 项目结构
```text
├── config.py             # 环境配置加载引擎
├── utils.py              # 核心架构组件与计算工具
├── 01_data_extractor.py  # 自动提取脚本
├── 01b_manual_corrector.py # 交互式修正工具
├── 02_plot_single_diagnostic.py # 单组诊断绘图
└── 03_plot_multi_comparison.py # 多组联合比对
```
