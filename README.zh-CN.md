# ZDEM Salt Kinematics

**ZDEM 盐构造运动学自动分析工具包**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

ZDEM 盐构造运动学自动分析工具包。

> 作者：[Phoenix0531-sudo](https://github.com/Phoenix0531-sudo) · 欢迎学习、二次开发与**商业使用**，请保留本仓库署名与许可证声明。

## 技术栈

Python · 科研分析

## 功能特性

- 数据提取与校正
- 单工况诊断图
- 多方案对比图

## 快速开始

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
```

```bash
pip install -e .
python 01_data_extractor.py
```

更完整的英文说明见 [README.md](README.md)。

## 仓库结构（摘要）

```
ZDEM_Salt_Kinematics/
├─ .github/
├─ copyright/
├─ demo_output/
├─ docs/
├─ zdem_salt_kinematics/
├─ zdem_salt_kinematics.egg-info/
├─ 01_data_extractor.py
├─ 01b_manual_corrector.py
├─ 02_plot_single_diagnostic.py
├─ 03_plot_multi_comparison.py
├─ CHANGELOG.md
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ README.zh-CN.md
├─ requirements.txt
├─ setup.py
```

## 测试

```bash
pip install pytest
pytest -q
```

仓库内 `tests/` 至少包含 smoke 测试；有完整测试套件时以 CI 为准。

## CI

GitHub Actions（`push` / `pull_request`）会：

- 安装依赖（requirements / pyproject）
- 运行 `pytest`（**硬失败**）
- 尽力做语法/结构检查

## 许可证

[MIT](LICENSE) — 可自由使用、修改、分发与**商用**，需保留版权与许可声明（提及本仓库 / 作者即可）。

## 关于

维护者：[Phoenix0531-sudo](https://github.com/Phoenix0531-sudo)
