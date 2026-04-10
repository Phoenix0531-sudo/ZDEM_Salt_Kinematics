"""
ZDEM Salt Kinematics 通用工具模块

职责: 提供通用的文件解析、数字信号处理和物理计算辅助函数。
"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Tuple, Optional

def setup_academic_style():
    """配置全项目统一的学术级绘图样式。"""
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'Arial', 'sans-serif'],
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

def extract_step_from_filename(filename: str) -> int:
    """从文件名中提取时间步编号。"""
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0

def apply_savgol_filter(data: np.ndarray, window_len: int, polyorder: int = 3) -> np.ndarray:
    """
    自适应窗口长度的 Savitzky-Golay 滤波器。
    
    Args:
        data: 原始信号。
        window_len: 理想窗口长度。
        polyorder: 多项式拟合阶数。
        
    Returns:
        平滑后的信号。
    """
    if data is None or len(data) < 3:
        return data if data is not None else np.array([])
    
    # 过滤掉 NaN 才能进行滤波
    mask = ~np.isnan(data)
    if np.sum(mask) < 3:
        return data
        
    actual_window = min(window_len, np.sum(mask))
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window <= polyorder:
        actual_window = polyorder + 1
        if actual_window % 2 == 0: actual_window += 1
        
    filtered_data = data.copy()
    filtered_data[mask] = savgol_filter(data[mask], window_length=actual_window, polyorder=polyorder)
    return filtered_data

def parse_zdem_dat_core(dat_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """
    底层 ZDEM .dat 格式解析引擎。
    
    Returns:
        (df_group, df_coord, right_wall_x)
    """
    group_data, coord_data = [], []
    group_cols, coord_cols = [], []
    in_group, in_coord, in_wall = False, False, False
    right_wall_x = None

    with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_lower = line.lower()
            
            # 墙体解析逻辑
            if 'p1[0]' in line_lower and 'p2[0]' in line_lower:
                in_wall = True
                continue
            if in_wall:
                parts = line.strip().split()
                if not parts or not parts[0].isdigit():
                    in_wall = False
                elif len(parts) >= 6 and parts[1] == '2':
                    right_wall_x = float(parts[2])
                    in_wall = False
                continue

            parts = line.strip().split()
            if not parts:
                in_group, in_coord = False, False
                continue

            # 组信息段
            if 'group' in line_lower and 'fric' in line_lower:
                in_group, in_coord = True, False
                group_cols = line_lower.split()
                continue
            # 坐标信息段
            elif 'rad' in line_lower and 'color' in line_lower and 'x' in line_lower:
                in_coord, in_group = True, False
                coord_cols = line_lower.split()
                continue

            # 数据行判断
            if parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                if in_group:
                    group_data.append(parts)
                elif in_coord:
                    coord_data.append(parts)

    df_g = pd.DataFrame(group_data)
    if not df_g.empty and group_cols:
        df_g.columns = group_cols[:len(df_g.columns)]

    df_c = pd.DataFrame(coord_data)
    if not df_c.empty and coord_cols:
        df_c.columns = coord_cols[:len(df_c.columns)]

    return df_g, df_c, right_wall_x
