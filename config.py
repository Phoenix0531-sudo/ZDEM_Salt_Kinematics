# pyright: reportMissingTypeStubs=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
"""
ZDEM Salt Kinematics 全局配置模块

职责: 从 .env 加载环境变量，执行类型转换与缺失校验，导出模块级常量。
.env 文件是唯一配置真理源，本模块不硬编码任何默认值。
"""
import os
import json
import sys
from typing import Any
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """读取必需环境变量，缺失时立即终止并报告。"""
    val = os.getenv(key)
    if val is None:
        print(f"[FATAL] 环境变量 '{key}' 未定义。请检查 .env 文件是否存在且包含该键。", file=sys.stderr)
        sys.exit(1)
    return val


# ==========================================
# 1. 核心目录与实验组配置
# ==========================================
FINAL_OUTPUT_DIR: str = _require('FINAL_OUTPUT_DIR')

# 实验组以 JSON 注入，解析后自动拼合 base_dir
_parsed_groups: list[dict[str, Any]] = json.loads(_require('EXPERIMENT_GROUPS'))
EXPERIMENT_GROUPS: list[dict[str, str]] = [
    {
        'base_dir': os.path.join(FINAL_OUTPUT_DIR, g['subdir']),
        'label': g['label'],
        'color': g['color'],
        'marker': g['marker'],
    }
    for g in _parsed_groups
]

CSV_FILENAME: str = _require('CSV_FILENAME')
PKL_FILENAME: str = _require('PKL_FILENAME')

# ==========================================
# 2. 物理模型尺寸与绘图视口配置
# ==========================================
MODEL_WIDTH: float = float(_require('MODEL_WIDTH'))
MODEL_HEIGHT: float = float(_require('MODEL_HEIGHT'))

MANUAL_PLOT_Y_MIN: float = float(_require('MANUAL_PLOT_Y_MIN'))
MANUAL_PLOT_Y_MAX: float = float(_require('MANUAL_PLOT_Y_MAX'))

MAX_SHORTENING_KM: float = float(_require('MAX_SHORTENING_KM'))
MAX_ASPECT_RATIO: float = float(_require('MAX_ASPECT_RATIO'))

# ==========================================
# 3. 离散元物理参数与算法阈值
# ==========================================
PUSHING_WALL_SIDE: str = _require('PUSHING_WALL_SIDE')
MIN_RELIEF_THRESHOLD: float = float(_require('MIN_RELIEF_THRESHOLD'))
SALT_GROUP_NAME: str = _require('SALT_GROUP_NAME')
PARTICLE_RADIUS: float = float(_require('PARTICLE_RADIUS'))
NUM_BINS: int = int(_require('NUM_BINS'))
SMOOTHING_WINDOW: int = int(_require('SMOOTHING_WINDOW'))
EXTRACT_SMOOTH_WINDOW: int = int(_require('EXTRACT_SMOOTH_WINDOW'))
PRE_EXTRUSION_FRAMES: int = int(_require('PRE_EXTRUSION_FRAMES'))
POST_EXTRUSION_FRAMES: int = int(_require('POST_EXTRUSION_FRAMES'))
FLANK_SLOPE_THRESHOLD: float = float(_require('FLANK_SLOPE_THRESHOLD'))

# 派生常量（由上述原子变量计算，非环境变量）
PARTICLE_AREA: float = np.pi * (PARTICLE_RADIUS ** 2)
