# ZDEM Salt Kinematics

**Salt geometry & kinematic metrics from ZDEM dumps**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Pipeline for **salt-tectonics** style ZDEM experiments: extract salt body geometry / evolution metrics from `.dat` dumps, optional manual correction, then single-run diagnostics and multi-run comparison plots.

## Why this exists

Salt kinematics papers need consistent extraction (peaks, thickness trends, filtered series) across many timesteps. Numbered scripts keep the order obvious.

## Features

- `01_data_extractor.py` — parallel-friendly extraction + logging
- `01b_manual_corrector.py` — human correction pass when auto geometry fails
- `02_plot_single_diagnostic.py` — one experiment diagnostic figures
- `03_plot_multi_comparison.py` — cross-experiment comparison
- Shared library under `zdem_salt_kinematics/`

## Install

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
pip install -r requirements.txt
pip install -e .
```

## Usage

Typical order:

```bash
python 01_data_extractor.py
python 01b_manual_corrector.py   # only if needed
python 02_plot_single_diagnostic.py
python 03_plot_multi_comparison.py
```

Configure paths and group definitions in `zdem_salt_kinematics/config.py` (and script constants) before running.

## Project layout

```
01_*.py 02_*.py 03_*.py
zdem_salt_kinematics/
demo_output/          # example figures when present
tests/
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
## License

MIT. Free for commercial use with attribution. See [LICENSE](LICENSE).
