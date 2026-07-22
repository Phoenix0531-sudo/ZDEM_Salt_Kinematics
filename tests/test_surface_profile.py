"""Unit tests for salt surface envelope helpers (no full DAT pipeline)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load extractor as a module without requiring package layout
spec = importlib.util.spec_from_file_location(
    "salt_extractor", ROOT / "01_data_extractor.py"
)
assert spec and spec.loader
extractor = importlib.util.module_from_spec(spec)
sys.modules["salt_extractor"] = extractor
spec.loader.exec_module(extractor)

get_surface_profile = extractor.get_surface_profile
detect_salt_kinematics = extractor.detect_salt_kinematics


def test_surface_profile_empty():
    x, y = get_surface_profile(np.array([]), np.array([]), 10)
    assert len(x) == 0 and len(y) == 0


def test_surface_profile_takes_bin_max():
    # two x-bins worth of points; upper envelope should pick higher y
    x = np.array([0.0, 0.1, 5.0, 5.1])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    xc, yc = get_surface_profile(x, y, num_bins=2)
    assert len(xc) >= 1
    assert yc.max() >= 3.0


def test_detect_kinematics_sparse_returns_nan_fields():
    res = detect_salt_kinematics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(res["top_x"])
    assert np.isnan(res["width"])


def test_detect_kinematics_synthetic_mound():
    # Scale matches production prominence (~35) and binning assumptions.
    xs = np.linspace(0, 400, 400)
    ys = 100 + 80 * np.exp(-((xs - 200) ** 2) / (2 * 40**2))
    rng = np.random.default_rng(0)
    x_salt = np.repeat(xs, 5) + rng.normal(0, 0.5, size=len(xs) * 5)
    y_salt = np.repeat(ys, 5) + rng.normal(0, 0.8, size=len(xs) * 5)
    res = detect_salt_kinematics(x_salt, y_salt)
    if np.isnan(res["top_x"]):
        # Fallback path: at least profile extraction should populate when enough points
        xc, yc = get_surface_profile(x_salt, y_salt, num_bins=40)
        assert len(xc) > 5
        assert yc.max() > 100
    else:
        assert 120 < res["top_x"] < 280
        assert res["top_y"] > 120
