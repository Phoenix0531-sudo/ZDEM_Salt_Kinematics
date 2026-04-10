"""
ZDEM Salt Kinematics 数据提取器

职责: 自动化提取 ZDEM 原始数据中的盐体几何形态与演化指标。
工程化改进: 接入统一日志系统，优化颗粒(Particle)相关术语，增强异常处理健壮性。
"""
import os
import pickle
import logging
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from tqdm import tqdm

from config import *
from utils import (
    extract_step_from_filename, 
    apply_savgol_filter, 
    parse_zdem_dat_core, 
    GroupDataManager,
    setup_project_logging
)

# 强制初始化项目级日志
setup_project_logging()

def get_surface_profile(x_all: np.ndarray, y_all: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过分箱统计提取模型表面包络线。
    
    Parameters
    ----------
    x_all : np.ndarray
        所有颗粒的水平坐标 (m)。
    y_all : np.ndarray
        所有颗粒的垂直坐标 (m)。
    num_bins : int
        分箱数量。
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        包络线采样点的 (x, y) 坐标数组。
    """
    if len(x_all) == 0:
        return np.array([]), np.array([])
    x_min, x_max = np.min(x_all), np.max(x_all)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    stat, bin_edges, _ = binned_statistic(x_all, y_all, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid = ~np.isnan(stat)
    return bin_centers[valid], stat[valid]

def detect_salt_kinematics(x_salt: np.ndarray, y_salt: np.ndarray) -> Dict[str, Any]:
    """
    核心演化算法：识别盐体颗粒群的主峰顶点与边界基点。
    采用 Prominence 峰值检测与局部坡度阈值扫描相结合的复合算法。
    """
    res = {'top_x': np.nan, 'top_y': np.nan, 'base_x': np.nan, 'base_y': np.nan, 'width': np.nan, 'relief': np.nan}
    if len(x_salt) < 10: 
        return res
        
    x_min, x_max = np.min(x_salt), np.max(x_salt)
    bins = np.linspace(x_min, x_max, NUM_BINS + 1)
    stat, bin_edges, _ = binned_statistic(x_salt, y_salt, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid = ~np.isnan(stat)
    x_prof, y_prof = bin_centers[valid], stat[valid]
    y_smooth = apply_savgol_filter(y_prof, EXTRACT_SMOOTH_WINDOW)
    
    # 峰值定位
    peaks, _ = find_peaks(y_smooth, prominence=35.0)
    if len(peaks) == 0: 
        return res
    peak_idx = peaks[np.argmax(y_smooth[peaks])]
    res['top_x'], res['top_y'] = x_prof[peak_idx], y_smooth[peak_idx]
    
    # 边界基点定位 (坡度阈值法)
    dy_dx = np.gradient(y_smooth, x_prof)
    abs_slope = np.abs(dy_dx)
    base_idx = 0
    if PUSHING_WALL_SIDE.lower() == 'right':
        # 从峰值向左扫描，寻找坡度趋于平缓且高度较低的点作为盐体边缘
        depth_limit = np.min(y_smooth[:peak_idx]) + 50.0
        scan_range = range(peak_idx - 5, -1, -1)
        for i in scan_range:
            if np.mean(abs_slope[i:i+3]) < FLANK_SLOPE_THRESHOLD and y_smooth[i] <= depth_limit:
                base_idx = i
                break
    
    res['base_x'], res['base_y'] = x_prof[base_idx], y_smooth[base_idx]
    res['relief'] = res['top_y'] - res['base_y']
    res['width'] = abs(res['top_x'] - res['base_x'])
    
    if res['relief'] < MIN_RELIEF_THRESHOLD:
        for k in res: res[k] = np.nan
    return res

def process_single_file(dat_path: str, initial_right_wall: Optional[float]) -> Tuple[int, Optional[Dict], Optional[Dict]]:
    """并行调用的原子任务：解析单帧 .dat 数据并提取核心运动学指标。"""
    step = extract_step_from_filename(dat_path)
    try:
        df_group, df_coord, right_wall_x = parse_zdem_dat_core(dat_path)
        if df_group.empty or df_coord.empty: 
            return step, None, None
        
        # 计算当前构造缩短量 (Shortening)
        shortening = (initial_right_wall - right_wall_x) if (right_wall_x is not None and initial_right_wall is not None) else np.nan
        
        # 数据对齐与类型安全转换
        df_coord.columns = [c.lower() for c in df_coord.columns]
        df_group.columns = [c.lower() for c in df_group.columns]
        for c in ['id', 'x', 'y']: 
            df_coord[c] = pd.to_numeric(df_coord[c], errors='coerce')
        df_group['id'] = pd.to_numeric(df_group['id'], errors='coerce')
        df_coord.dropna(subset=['id', 'x', 'y'], inplace=True)
        
        # 关联组别信息，过滤目标颗粒
        df_merged = pd.merge(df_coord, df_group[['id', 'group']], on='id')
        salt_df = df_merged[df_merged['group'].astype(str).str.lower() == SALT_GROUP_NAME]
        
        # 1. 盐体颗粒出露面积计算逻辑
        # 原理: 提取全模型颗粒形成的表面包络线，识别高出包络线一定阈值的盐体颗粒并累加横截面积
        x_surf_all, y_surf_all = get_surface_profile(df_coord['x'].values, df_coord['y'].values, NUM_BINS)
        extruded_area = 0.0
        if not salt_df.empty and len(x_surf_all) > 0:
            surf_y_interp = np.interp(salt_df['x'].values, x_surf_all, y_surf_all)
            # 判定阈值：高出表面约 1.5 倍颗粒半径 (PARTICLE_RADIUS)
            extruded_mask = salt_df['y'].values >= (surf_y_interp - 1.5 * PARTICLE_RADIUS)
            extruded_area = np.sum(extruded_mask) * PARTICLE_AREA

        # 2. 核心几何运动学检测
        kin = detect_salt_kinematics(salt_df['x'].values, salt_df['y'].values) if not salt_df.empty else {}
        
        res_row = {
            'Step': step, 
            'Actual_Shortening': shortening, 
            'Extruded_Area': extruded_area,
            'Width': kin.get('width', np.nan), 
            'Relief': kin.get('relief', np.nan),
            'Aspect_Ratio': (kin['relief'] / kin['width']) if kin.get('width', 0) > 0 else np.nan
        }
        
        profile_cache = {
            'step': step, 
            'top_x': kin.get('top_x', np.nan), 'top_y': kin.get('top_y', np.nan),
            'base_x': kin.get('base_x', np.nan), 'base_y': kin.get('base_y', np.nan),
            'x': np.array([]), 'y': np.array([])
        }
        
        if not salt_df.empty:
            xb = np.linspace(np.min(salt_df['x']), np.max(salt_df['x']), NUM_BINS)
            yb, _, _ = binned_statistic(salt_df['x'], salt_df['y'], statistic='max', bins=len(xb))
            profile_cache['x'], profile_cache['y'] = xb, yb
            
        return step, res_row, profile_cache

    except Exception as e:
        logging.error(f"处理单帧数据失败 [Step {step}]: {e}")
        return step, None, None

def main():
    """主程序入口：遍历实验组并执行并行提取流程。"""
    if not os.path.exists(FINAL_OUTPUT_DIR): 
        os.makedirs(FINAL_OUTPUT_DIR)
        
    for group in EXPERIMENT_GROUPS:
        mgr = GroupDataManager(group)
        dat_files = mgr.get_dat_files()
        if not dat_files: 
            logging.warning(f"实验组 [{mgr.label}] 目录中未检测到有效的 .dat 文件，跳过处理。")
            continue
        
        # 初始状态提取，作为后续缩短量计算的零点参考
        _, _, initial_wall = parse_zdem_dat_core(dat_files[0])
        results, profiles = [], {}
        
        # 启用多进程并行加速解析
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f, initial_wall) for f in dat_files]
            pbar = tqdm(
                concurrent.futures.as_completed(futures), total=len(dat_files),
                desc=f"提取进度 [{mgr.label}]", unit="帧",
                bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            for f in pbar:
                step, res, prof = f.result()
                if res: 
                    results.append(res)
                    profiles[step] = prof

        # 数据持久化与平滑后处理
        if not results:
            logging.error(f"实验组 [{mgr.label}] 无有效提取产物，请检查原始数据。")
            continue
            
        df = pd.DataFrame(results).sort_values('Step').reset_index(drop=True)
        # 统一缩短量单位为 km
        df['Shortening_km'] = df['Actual_Shortening'].interpolate() / 1000.0
        
        # 执行时间序列平滑（去除模拟过程中的物理震荡噪声）
        for col in ['Aspect_Ratio', 'Width', 'Relief']:
            if col in df.columns:
                df[f'{col}_Smooth'] = df[col].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()

        # 导出工程化 CSV 指标与可视化缓存 PKL
        df.to_csv(mgr.csv_path, index=False)
        with open(mgr.pkl_path, 'wb') as f: 
            pickle.dump(profiles, f)
            
        logging.info(f"实验组 [{mgr.label}] 颗粒演化指标分析流程执行完毕。")

if __name__ == '__main__':
    main()
