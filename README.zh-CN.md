# ZDEM Salt Kinematics

**从 ZDEM 结果提取盐构造几何与运动学指标 — 表面包络、边缘向斜式基点、多模型对比图。**

[English](README.md) | [中文](README.zh-CN.md)

[![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

盐构造 DEM 后处理工具。与 [ParticleTracker](https://github.com/Phoenix0531-sudo/ZDEM_ParticleTracker)（交互点选）配套：本仓侧重 **批量几何提取与论文向出图**。

## 预览

![ZDEM Salt Kinematics](docs/screenshots/preview.png)

## 流水线

| 脚本 | 作用 |
|------|------|
| `01_data_extractor.py` | 并行解析帧 → 运动学表 / pickle |
| `01b_manual_corrector.py` | 自动峰失败时的人工修正 |
| `02_plot_single_diagnostic.py` | 单次诊断图 |
| `03_plot_multi_comparison.py` | 多模型对比 |
| `zdem_salt_kinematics/` | config、utils（解析、滤波、日志、分组） |

核心：`get_surface_profile`（分箱 max 包络）、`detect_salt_kinematics`（顶/底/宽/relief）、`parse_zdem_dat_core`、并发 + tqdm。

## 安装运行

```bash
pip install -e .
python 01_data_extractor.py
pytest tests/
```

参数见 `zdem_salt_kinematics/config.py`。

## 许可证

MIT。详见 [LICENSE](LICENSE)。
