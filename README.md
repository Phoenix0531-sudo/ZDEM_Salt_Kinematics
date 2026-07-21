<div align="center">

# ZDEM Salt Kinematics · 盐体运动学分析系统

**Automated kinematic analysis toolkit for ZDEM salt tectonics simulations**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)

**Automated Kinematic Analysis Toolkit for ZDEM Salt Tectonics Simulations**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/Numpy-%3E%3D1.21-013243?logo=numpy)]()
[![SciPy](https://img.shields.io/badge/SciPy-%3E%3D1.7-8CAAE6?logo=scipy)]()

</div>

---

## 项目简介 | Overview

本系统专为 ZDEM (Zhong-Discrete Element Method) 离散元模拟数据设计，用于自动化提取盐体（Salt Body）在构造缩短过程中的几何形态演化指标。系统采用分箱统计与自适应 Savitzky-Golay 平滑算法，从数万颗粒的原始数据中稳健识别盐丘主峰与边缘基点，计算宽度（Width）、起伏（Relief）与高宽比（Aspect Ratio）三大核心运动学指标，并生成符合学术出版标准的诊断图表。

> This toolkit is designed for ZDEM (Zhong-Discrete Element Method) particle simulation data, providing automated extraction of geometric evolution indicators for salt bodies during tectonic shortening. Using binned statistics and adaptive Savitzky-Golay filtering, it robustly identifies the salt diapir crest and rim syncline base from raw particle data, computes three key kinematic metrics (width, relief, aspect ratio), and generates publication-ready diagnostic figures.

---

## 技术特性 | Technical Highlights

| 特性 | Feature | 说明 |
|------|---------|------|
| **全自动提取** | Fully Automated Extraction | 批量解析 .dat 原始粒子数据，自动计算运动学关键点无人工干预 |
| **专家级质控** | Expert QA/QC Interface | 基于 Matplotlib 的交互式 UI，支持鼠标点击手动校准算法误判的基点位置 |
| **Rio Tinto 基点算法** | Pit-detect Base Algorithm | 融合坑底追踪与反弹判定机制，精准识别边缘向斜（Rim Syncline）边界 |
| **学术级绘图** | Publication-ready Figures | 600 DPI 高精度输出，Y 轴右置学术风格，出露前实线 + 出露后虚线标识 |
| **并发处理引擎** | Concurrent Processing | 基于 ProcessPoolExecutor 的多进程并行解析，自动管理大规模实验组比对 |
| **全管线平滑一致性** | Consistent Smoothing Pipeline | 统一的空间维 SavGol 与时间维滚动平均滤波，确保提取-修正-渲染全链路一致性 |

---

## 目录 | Table of Contents

- [数据准备 / Data Preparation](#数据准备--data-preparation)
- [算法原理 / Algorithm](#算法原理--algorithm)
- [模块文档 / Module Reference](#模块文档--module-reference)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [输出说明 / Output](#输出说明--output)
- [安装依赖 / Installation](#安装依赖--installation)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [引用 / Citation](#引用--citation)
- [许可证 / License](#许可证--license)

---

## 数据准备 | Data Preparation

用户在运行本系统之前，需要确保 ZDEM 模拟数据已按实验组整理为以下目录结构：

```
FINAL_OUTPUT_DIR/
  ├── GroupA/
  │   └── data/
  │       ├── 0001.dat
  │       ├── 0002.dat
  │       └── ...
  ├── GroupB/
  │   └── data/
  │       ├── 0001.dat
  │       └── ...
  └── ...
```

每个实验组目录下需包含 `data/` 子文件夹，存放按时间步编号的 `.dat` 原始粒子数据文件。系统通过 `.env` 配置文件中的 `EXPERIMENT_GROUPS` JSON 数组声明各组路径、标签与可视化样式。

> Before running the system, organize your ZDEM simulation data into experiment groups with the directory structure above. Each group folder must contain a `data/` subdirectory with timestep-numbered `.dat` files. The `.env` configuration file declares group paths, labels, and visual styles via the `EXPERIMENT_GROUPS` JSON array.

---

## 算法原理 | Algorithm

本系统的核心运动学分析算法包含三个环节：

1. **表面包络线提取**：对粒子云进行水平向分箱（`scipy.stats.binned_statistic`），取每个箱体内的最大高程值作为上包络线，得到盐体剖面轮廓。

2. **特征点定位**：
   - **主峰（Crest/Top)**：对平滑后的包络线应用 `scipy.signal.find_peaks`，结合 Prominence 阈值筛选最显著峰值作为盐丘顶点。
   - **基点（Base/Rim Syncline)**：采用坑底探测 + 反弹判定算法，从峰顶向推板反方向扫描，追踪局部极小值点，当连续反弹超过 `PATIENCE` 帧且上升高度超过 `BOUNCE_MIN_HEIGHT` 时锁定基点位。辅以局部斜率阈值（`FLANK_SLOPE_THRESHOLD`）作为终止条件。

3. **数据平滑**：空间维采用自适应窗口的 Savitzky-Golay 滤波器（窗口长度自动适配有效数据点数），时间维采用中心化滚动平均，确保演化曲线平滑无假频。

> The core kinematic analysis algorithm consists of three stages. First, surface envelope extraction via horizontal binning (`scipy.stats.binned_statistic`) with maximum elevation sampling. Second, feature point localization: crest detection using `find_peaks` with prominence filtering, and base detection via a pit-tracking with bounce-validation algorithm that scans outward from the crest. Third, dual-domain smoothing with adaptive Savitzky-Golay filtering (spatial) and centered rolling average (temporal).

---

## 模块文档 | Module Reference

### 核心模块 | Core Modules

| 模块 | Module | 功能 | 导出常量/类 |
|------|--------|------|-------------|
| 配置引擎 | `config.py` | 从 `.env` 加载环境变量，执行类型转换与缺失校验 | 全部大写全局常量 |
| 通用工具 | `utils.py` | 提供共享的物理计算、文件解析、绘图样式与日志支持 | `GroupDataManager`, `ProfileData`, `extract_step_from_filename`, `apply_savgol_filter`, `parse_zdem_dat_core`, `setup_project_logging`, `setup_academic_style` |

### 配置参数 | Configuration Parameters

| 参数 | Parameter | 类型 | 说明 |
|------|-----------|------|------|
| `FINAL_OUTPUT_DIR` | Output root | str | 数据存放的总根目录路径 |
| `EXPERIMENT_GROUPS` | Group config | JSON | 实验组定义，含子目录、标签、颜色与标记 |
| `NUM_BINS` | Sampling bins | int | 剖面包络线的水平采样分辨率 (默认 300) |
| `PUSHING_WALL_SIDE` | Wall side | str | 挤压墙方位，left 或 right |
| `SMOOTHING_WINDOW` | Smooth window | int | 时间序列滚动平均平滑跨度 (默认 3) |
| `EXTRACT_SMOOTH_WINDOW` | Extract smooth | int | 空间包络线 SavGol 平滑窗口 (默认 31) |
| `FLANK_SLOPE_THRESHOLD` | Slope threshold | float | 基点识别局部坡度阈值 (默认 0.15) |
| `MIN_RELIEF_THRESHOLD` | Relief threshold | float | 有效变形最小隆起高度 (默认 100.0 m) |
| `PARTICLE_RADIUS` | Particle radius | float | 离散元颗粒平均物理半径 (默认 70.0 m) |

### 数据脚本 | Data Scripts

| 脚本 | Script | 功能 |
|------|--------|------|
| `01_data_extractor.py` | Data Extractor | 批量解析 .dat 文件，提取运动学指标并生成 CSV/PKL 缓存 |
| `01b_manual_corrector.py` | Manual Corrector | 交互式 QA/QC 工具，支持鼠标点击修正基点坐标 |
| `02_plot_single_diagnostic.py` | Single Diagnostic Plot | 生成单组实验的演化曲线与剖面形态诊断矩阵 |
| `03_plot_multi_comparison.py` | Multi-group Comparison | 生成所有实验组的运动学指标联合对比图（PNG+PDF） |

---

## 快速开始 | Quick Start

### 环境准备 | Setup

```bash
pip install -r requirements.txt
```

将 `.env.example` 复制为 `.env`，修改其中的目录路径与实验组配置以匹配本地数据。

> Copy `.env.example` to `.env` and modify the directory paths and experiment group configuration to match your local data.

### 标准工作流 | Standard Workflow

**Step 1 — 数据提取 / Data Extraction**

```bash
python 01_data_extractor.py
```

批量解析各组 `.dat` 文件，自动计算每帧的运动学指标，生成采样后的 CSV 与剖面缓存 PKL 文件。

**Step 2 — 人工质控 / QA/QC Correction**

```bash
python 01b_manual_corrector.py
```

交互式检查每帧特征点定位结果。鼠标点击图中盐体表面可重定位基点，快捷键 [S] 或 [Enter] 保存修正并退出。

**Step 3 — 单组诊断图 / Single-group Diagnostic**

```bash
python 02_plot_single_diagnostic.py
```

为每个实验组输出运动学演化轨迹图（`Kinematic_Evolution_Diagnostic.png`）与盐体剖面多宫格诊断图（`Salt_Profiles_Diagnostic_Grid.png`）。

**Step 4 — 跨组对比 / Multi-group Comparison**

```bash
python 03_plot_multi_comparison.py
```

生成所有实验组的宽度、起伏、高宽比联合对比图，同时输出高分辨率 PNG 与矢量 PDF 格式。

---

## 输出说明 | Output

### 控制台输出 | Console Output

系统启动后，日志以 `HH:MM:SS [LEVEL] message` 格式输出，包含每组处理的文件数量、采样进度与关键指标变化。

```
05:23:47 [INFO] 正在处理实验组: GroupA (共 850 帧)
05:25:12 [INFO] 组别 [GroupA] 完成采样与提取: 原始 850 帧 -> 采样 35 帧
05:25:12 [INFO] 数据已存入 -> .../GroupA/kinematics_data.csv | .../GroupA/profiles_cache.pkl
```

### 图表输出 | Figure Output

| 文件 | File | 说明 |
|------|------|------|
| `Kinematic_Evolution_Diagnostic.png` | Single Evolution | 单组高宽比演化轨迹 (600 DPI) |
| `Salt_Profiles_Diagnostic_Grid.png` | Profile Grid | 单组剖面形态多宫格诊断图 (300 DPI) |
| `Multi_Evolution_HalfWidth.png/pdf` | Width Comparison | 跨组半宽对比图 |
| `Multi_Evolution_Relief.png/pdf` | Relief Comparison | 跨组起伏对比图 |
| `Multi_Evolution_AspectRatio.png/pdf` | Aspect Ratio Comparison | 跨组高宽比对比图 |

---

## 安装依赖 | Installation

### 系统要求 | System Requirements

- Python 3.9 or later
- Windows / Linux / macOS

### 依赖列表 | Dependencies

| 包 | Package | 最低版本 | 用途 |
|---|---------|----------|------|
| NumPy | numpy | >= 1.21.0 | 数值计算与数组操作 |
| Pandas | pandas | >= 1.3.0 | 结构化数据管理与持久化 |
| Matplotlib | matplotlib | >= 3.4.0 | 学术级可视化渲染 |
| SciPy | scipy | >= 1.7.0 | 分箱统计、峰值检测与 SavGol 滤波 |
| tqdm | tqdm | >= 4.60.0 | 进度条显示 |
| python-dotenv | python-dotenv | >= 0.19.0 | 环境变量注入 |

```bash
pip install -r requirements.txt
```

---

## 项目结构 | Project Structure

```
ZDEM_Salt_Kinematics/
  ├── zdem_salt_kinematics/     # 核心包 (可 pip install 安装)
  │   ├── __init__.py           # 包元数据 (版本号、许可证)
  │   ├── config.py             # .env 配置加载引擎
  │   └── utils.py              # 类型定义、计算引擎与绘图样式
  ├── 01_data_extractor.py      # Step 1: 自动提取脚本
  ├── 01b_manual_corrector.py   # Step 2: 交互式 QA/QC 修正工具
  ├── 02_plot_single_diagnostic.py  # Step 3: 单组诊断绘图
  ├── 03_plot_multi_comparison.py   # Step 4: 多组联合比对
  ├── .env.example              # 环境配置模板
  ├── requirements.txt          # Python 依赖清单
  ├── setup.py                  # 包安装配置
  ├── LICENSE                   # MIT 开源许可证
  └── README.md                 # 本文档
```

---

## 引用 | Citation

If you use ZDEM Salt Kinematics in your research, please cite it as:

```bibtex
@software{zdem_salt_kinematics2026,
  title     = {{ZDEM Salt Kinematics}: Automated Kinematic Analysis Toolkit for ZDEM Salt Tectonics Simulations},
  year      = {2026},
  url       = {https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics},
  version   = {1.0.0},
  license   = {MIT}
}
```

---

## 许可证 | License

This project is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Made for the structural geology and salt tectonics research community**

</div>
