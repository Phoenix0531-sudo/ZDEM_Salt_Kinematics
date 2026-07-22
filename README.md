# ZDEM Salt Kinematics

**Salt geometry & kinematics metrics from ZDEM dumps — surface envelope, rim-syncline style base points, multi-step comparison plots.**

[English](README.md) | [中文](README.zh-CN.md)

[![CI](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Post-process toolkit for **salt-tectonics DEM campaigns**. Companion to [ZDEM Particle Tracker](https://github.com/Phoenix0531-sudo/ZDEM_ParticleTracker) (interactive picking) — this repo focuses on **batch geometry extraction and publication-oriented figures**.

## Preview

![ZDEM Salt Kinematics](docs/screenshots/preview.png)

## Pipeline (real scripts)

| Script | Role |
|--------|------|
| `01_data_extractor.py` | Parallel parse of ZDEM frames → kinematics table / pickle |
| `01b_manual_corrector.py` | Manual correction pass when auto peaks fail |
| `02_plot_single_diagnostic.py` | Single-run diagnostic figures |
| `03_plot_multi_comparison.py` | Multi-model comparison panels |
| `zdem_salt_kinematics/` | `config.py`, `utils.py` (parse core, Savitzky–Golay, logging, group manager) |

### Core algorithms in `01_data_extractor.py`

- `get_surface_profile(x, y, num_bins)` — `scipy.stats.binned_statistic(..., statistic='max')` for a surface envelope
- `detect_salt_kinematics(x_salt, y_salt)` — peak / rim-syncline oriented metrics: `top_x/y`, `base_x/y`, `width`, `relief`, profile arrays
- Uses `find_peaks`, optional smoothing via `apply_savgol_filter`
- `parse_zdem_dat_core` + `GroupDataManager` for color/group oriented particle subsets
- `concurrent.futures` + `tqdm` for multi-frame extraction
- Project logging via `setup_project_logging()`

Outputs land under configurable dirs / `demo_output/` samples.

## Install

```bash
git clone https://github.com/Phoenix0531-sudo/ZDEM_Salt_Kinematics.git
cd ZDEM_Salt_Kinematics
pip install -e .
# or: uv sync
# deps: numpy, pandas, matplotlib, scipy, tqdm, python-dotenv
```

Python **>= 3.9**.

## Run

```bash
python 01_data_extractor.py
python 02_plot_single_diagnostic.py
python 03_plot_multi_comparison.py
pytest tests/
```

Tune paths and binning in `zdem_salt_kinematics/config.py` / `.env` style settings.

## Related

ParticleTracker (interactive) · Area Conservation · Bond Fracture · DFN · Model Editor

## Scope

- **In:** salt body envelope metrics, batch extraction, paper-style plots
- **Out:** live GUI picking (use ParticleTracker), full rheology inversion

## License

MIT. See [LICENSE](LICENSE).
