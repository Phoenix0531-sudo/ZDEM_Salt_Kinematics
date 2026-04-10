"""
ZDEM Salt Kinematics 通用工具模块

职责: 提供全项目共享的物理计算、文件解析、学术绘图样式及日志系统支持。
工程化改进: 完善类型提示、优化颗粒(Particle)相关术语、升级日志格式。
"""
import re
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Any, Dict, List, TypedDict
from config import CSV_FILENAME, PKL_FILENAME

# ==========================================
# 1. 类型定义 (Type Definitions)
# ==========================================

class ProfileData(TypedDict):
    """
    单个时间步的盐体剖面缓存数据。
    
    Attributes
    ----------
    step : int
        时间步编号。
    x : np.ndarray
        水平向坐标数组 (m)。
    y : np.ndarray
        包络线高程数组 (m)。
    top_x : float
        主峰顶点水平坐标。
    top_y : float
        主峰顶点高程。
    base_x : float
        识别到的基点（边缘）水平坐标。
    base_y : float
        识别到的基点（边缘）高程。
    """
    step: int
    x: np.ndarray
    y: np.ndarray
    top_x: float
    top_y: float
    base_x: float
    base_y: float

# ==========================================
# 2. 核心架构组件 (Core Architecture)
# ==========================================

class GroupDataManager:
    """
    实验组数据资产管理器。
    
    职责: 统一管理实验组的输入（.dat）、中间缓存（.pkl）与指标产物（.csv）路径。
    """
    def __init__(self, group_config: Dict[str, str]):
        self.config = group_config
        self.label = group_config['label']
        self.base_dir = group_config['base_dir']
        # ZDEM 原始数据通常存放于各组目录下的 data 子文件夹
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.csv_path = os.path.join(self.base_dir, CSV_FILENAME)
        self.pkl_path = os.path.join(self.base_dir, PKL_FILENAME)

    def get_dat_files(self) -> List[str]:
        """获取并按时间步排序的所有原始数据文件。"""
        import glob
        pattern = os.path.join(self.data_dir, '*.dat')
        files = glob.glob(pattern)
        return sorted(files, key=extract_step_from_filename)

def setup_project_logging(level=logging.INFO):
    """
    配置全项目统一的工业级日志格式。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def setup_academic_style():
    """
    配置全项目统一的学术级绘图样式 (Publication-ready)。
    优化了中文字体兼容性、轴线粗细及刻度方向。
    """
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'sans-serif'],
        'axes.unicode_minus': False,
        'axes.linewidth': 1.2,
        'font.size': 11,
        'axes.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': False,
        'figure.dpi': 150
    })

# ==========================================
# 3. 计算与解析引擎 (Processing Engines)
# ==========================================

def extract_step_from_filename(filename: str) -> int:
    """
    从 ZDEM 文件名中提取时间步编号。
    """
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0

def apply_savgol_filter(data: Any, window_len: int, polyorder: int = 3) -> np.ndarray:
    """
    稳健的 Savitzky-Golay 平滑滤波器。
    针对 DEM 离散数据可能存在的 NaN 或点数不足情况进行了保护。
    """
    if data is None:
        return np.array([])
    
    arr = np.atleast_1d(np.asarray(data))
    if arr.size < 3:
        return arr

    mask = ~np.isnan(arr)
    if np.sum(mask) <= polyorder:
        return arr
        
    actual_window = min(window_len, np.sum(mask))
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window <= polyorder:
        actual_window = polyorder + 1
        if actual_window % 2 == 0: actual_window += 1
        
    filtered_data = arr.copy()
    try:
        filtered_data[mask] = savgol_filter(arr[mask], window_length=actual_window, polyorder=polyorder)
    except Exception:
        pass
    return filtered_data

def parse_zdem_dat_core(dat_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """
    底层 ZDEM .dat 原始文本解析引擎。
    
    职责: 高效解析颗粒组别映射、颗粒物理属性坐标及推板(Wall)实时位置。
    """
    group_data, coord_data = [], []
    group_cols, coord_cols = [], []
    in_group_block, in_coord_block, in_wall_block = False, False, False
    wall_x = None

    try:
        with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line:
                    continue
                
                line_lower = raw_line.lower()

                # --- 1. 动态边界 (墙体) 位置扫描 ---
                if 'p1[0]' in line_lower and 'p2[0]' in line_lower:
                    in_wall_block = True
                    continue
                
                if in_wall_block:
                    parts = raw_line.split()
                    if parts and parts[0].isdigit():
                        # 默认识别 ID 为 2 的墙体作为右侧推板
                        if len(parts) >= 3 and parts[1] == '2':
                            try:
                                wall_x = float(parts[2])
                            except ValueError:
                                pass
                            in_wall_block = False
                    else:
                        in_wall_block = False
                    continue

                # --- 2. 颗粒属性块识别 ---
                if 'group' in line_lower and 'fric' in line_lower:
                    in_group_block, in_coord_block = True, False
                    group_cols = line_lower.split()
                    continue
                elif 'rad' in line_lower and 'color' in line_lower and 'x' in line_lower:
                    in_coord_block, in_group_block = True, False
                    coord_cols = line_lower.split()
                    continue

                # 数据行采集
                if raw_line[0].isdigit() or (raw_line[0] == '-' and raw_line[1].isdigit()):
                    parts = raw_line.split()
                    if in_group_block:
                        group_data.append(parts)
                    elif in_coord_block:
                        coord_data.append(parts)
                else:
                    in_group_block = in_coord_block = False

        # 构建类型安全的 DataFrame
        df_group = pd.DataFrame(group_data)
        if not df_group.empty and group_cols:
            df_group.columns = group_cols[:len(df_group.columns)]
        
        df_coord = pd.DataFrame(coord_data)
        if not df_coord.empty and coord_cols:
            df_coord.columns = coord_cols[:len(df_coord.columns)]
            
        return df_group, df_coord, wall_x

    except Exception as e:
        logging.error(f"解析原始文件失败 [{os.path.basename(dat_path)}]: {e}")
        return pd.DataFrame(), pd.DataFrame(), None
