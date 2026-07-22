# ZDEM Salt Kinematics

**Salt geometry & kinematics metrics from ZDEM — publication-grade plots.**

[English](README.md) | [中文](README.zh-CN.md)

[![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Salt geometry & kinematics metrics from ZDEM — publication-grade plots.

Geometry evolution. Lab-report ready figures.


## Features

- 🧂 Salt kinematics oriented metrics
- 🖼️ Publication-grade plotting helpers
- 📦 Pure `pyproject` packaging + UV lock CI
- 🧪 Real pytest smoke

## Get started

### Install

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
uv sync   # or pip install -e .
```

### Usage

```bash
python -m zdem_salt_kinematics --help  # if console script present
pytest tests/
```

## Project layout

```
zdem_salt_kinematics/
demo_output/  tests/  docs/
```

## Related ZDEM tools

| Repo | Role |
|------|------|
| [ZDEM_ParticleTracker](https://github.com/Phoenix0531-sudo/ZDEM_ParticleTracker) | Interactive particle tracking + VisPy true-radius render |
| [ZDEM_Salt_Kinematics](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics) | Salt geometry / kinematics extraction & plots |
| [ZDEM_Area_Conservation](https://github.com/Phoenix0531-sudo/ZDEM_Area_Conservation) | Area-conservation / triangulation analysis |
| [ZDEM_Bond_Fracture](https://github.com/Phoenix0531-sudo/ZDEM_Bond_Fracture) | Bond damage series + desktop / CLI |
| [ZDEM_Damage_Thresholds](https://github.com/Phoenix0531-sudo/ZDEM_Damage_Thresholds) | Damage thresholds & strain–energy plots |
| [ZDEM_DFN](https://github.com/Phoenix0531-sudo/ZDEM_DFN) | Discrete fracture network generator for ZDEM |
| [ZDEM_Model_Editor](https://github.com/Phoenix0531-sudo/ZDEM_Model_Editor) | Model file visual editor |
| [ZDEM_Archiver](https://github.com/Phoenix0531-sudo/ZDEM_Archiver) | Purge / archive bulky simulation dumps |
| [ZDEM3D_WEB](https://github.com/Phoenix0531-sudo/ZDEM3D_WEB) | CAE cloud UI (Django + React + VTK.js) |

## Notes

Companion to ParticleTracker for post-process kinematics, not interactive picking.

## License

MIT. Free for commercial use with attribution where applicable. See [LICENSE](LICENSE).
