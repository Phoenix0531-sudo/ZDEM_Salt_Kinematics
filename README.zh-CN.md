# ZDEM Salt Kinematics

**从 ZDEM 结果提取盐体几何与运动学指标**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

面向**盐构造**类 ZDEM 实验的流水线：从 `.dat` 提取盐体几何 / 演化指标，必要时人工校正，再出单次诊断图与多工况对比图。

## 为什么做这个

盐体运动学写作需要在大量时步上稳定提取（峰、厚度趋势、滤波序列）。编号脚本让顺序一目了然。

## 功能

- `01_data_extractor.py` — 可并行提取 + 日志  
- `01b_manual_corrector.py` — 自动几何失败时的人工修正  
- `02_plot_single_diagnostic.py` — 单实验诊断图  
- `03_plot_multi_comparison.py` — 跨实验对比  
- 共享库 `zdem_salt_kinematics/`  

## 安装

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
pip install -r requirements.txt
pip install -e .
```

## 使用

推荐顺序：

```bash
python 01_data_extractor.py
python 01b_manual_corrector.py   # 仅必要时
python 02_plot_single_diagnostic.py
python 03_plot_multi_comparison.py
```

运行前在 `zdem_salt_kinematics/config.py`（及脚本常量）中配置路径与组定义。

## 目录结构

```
01_*.py 02_*.py 03_*.py
zdem_salt_kinematics/
demo_output/
tests/
```

## 相关 ZDEM 工具

| 仓库 | 作用 |
|------|------|
| [ZDEM_ParticleTracker](https://github.com/Phoenix0531-sudo/ZDEM_ParticleTracker) | 交互式颗粒追踪 + VisPy 真实半径渲染 |
| [ZDEM_Salt_Kinematics](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics) | 盐体几何/运动学提取与出图 |
| [ZDEM_Area_Conservation](https://github.com/Phoenix0531-sudo/ZDEM_Area_Conservation) | 面积守恒 / 三角网格分析 |
| [ZDEM_Bond_Fracture](https://github.com/Phoenix0531-sudo/ZDEM_Bond_Fracture) | 粘结损伤序列 + 桌面/CLI |
| [ZDEM_Damage_Thresholds](https://github.com/Phoenix0531-sudo/ZDEM_Damage_Thresholds) | 损伤阈值与应变–能量图 |
| [ZDEM_DFN](https://github.com/Phoenix0531-sudo/ZDEM_DFN) | ZDEM 离散裂隙网络生成 |
| [ZDEM_Model_Editor](https://github.com/Phoenix0531-sudo/ZDEM_Model_Editor) | 模型文件可视化编辑 |
| [ZDEM_Archiver](https://github.com/Phoenix0531-sudo/ZDEM_Archiver) | 大体量模拟结果归档清理 |
| [ZDEM3D_WEB](https://github.com/Phoenix0531-sudo/ZDEM3D_WEB) | CAE 云端界面（Django + React + VTK.js） |
## 许可证

MIT。可在署名前提下商用。见 [LICENSE](LICENSE)。
