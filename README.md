# ZDEM Salt Kinematics

**Salt geometry and kinematics metrics from ZDEM with publication-grade plots.**

[English](README.md) | [中文](README.zh-CN.md)

[![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Post-process metrics for lab reports and papers.

## Preview

![ZDEM Salt Kinematics](docs/screenshots/preview.png)

## Features

- Salt kinematics oriented metrics
- Publication-grade plotting helpers
- Pure pyproject packaging + UV lock CI
- Real pytest smoke

## Get started

### Install

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
uv sync
```

### Usage

```bash
pytest tests/
```

## Project layout

```
zdem_salt_kinematics/
demo_output/  tests/
```

## Related ZDEM tools

| Repo | Role |
|------|------|
| [ZDEM_ParticleTracker](https://github.com/Phoenix0531-sudo/ZDEM_ParticleTracker) | Interactive particle tracking + true-radius render |
| [ZDEM_Salt_Kinematics](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics) | Salt geometry / kinematics extraction and plots |
| [ZDEM_Area_Conservation](https://github.com/Phoenix0531-sudo/ZDEM_Area_Conservation) | Area-conservation / triangulation analysis |
| [ZDEM_Bond_Fracture](https://github.com/Phoenix0531-sudo/ZDEM_Bond_Fracture) | Bond damage series + visualizer |
| [ZDEM_Damage_Thresholds](https://github.com/Phoenix0531-sudo/ZDEM_Damage_Thresholds) | Damage thresholds and energy plots |
| [ZDEM_DFN](https://github.com/Phoenix0531-sudo/ZDEM_DFN) | Discrete fracture network generator |
| [ZDEM_Model_Editor](https://github.com/Phoenix0531-sudo/ZDEM_Model_Editor) | Model file visual editor |
| [ZDEM_Archiver](https://github.com/Phoenix0531-sudo/ZDEM_Archiver) | Archive / purge bulky dumps |
| [ZDEM3D_WEB](https://github.com/Phoenix0531-sudo/ZDEM3D_WEB) | CAE cloud UI (Django + React + VTK.js) |


## Notes

Companion to ParticleTracker for post-process kinematics.

## License

MIT. Free for commercial use with attribution where applicable. See [LICENSE](LICENSE).
