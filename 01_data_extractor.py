"""
ZDEM Salt Kinematics 数据提取器 (V2.0)

职责: 自动化批处理 ZDEM 离散元模拟输出，提取盐丘运动学特征。
工程化改进: 模块化函数设计、多进程并发加速、利用 utils 通用解析引擎。
"""
import os
import glob
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
from utils import extract_step_from_filename, apply_savgol_filter, parse_zdem_dat_core

# ==========================================
# 1. 核心计算子函数
# ==========================================

def get_surface_profile(x_all: np.ndarray, y_all: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """利用分箱统计计算包络地表剖面。"""
    x_min, x_max = np.min(x_all), np.max(x_all)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    stat, bin_edges, _ = binned_statistic(x_all, y_all, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid = ~np.isnan(stat)
    return bin_centers[valid], stat[valid]

def detect_salt_kinematics(x_salt: np.ndarray, y_salt: np.ndarray) -> Dict[str, Any]:
    """
    分析盐层颗粒分布，提取宽度、起伏度和锚点。
    
    Returns:
        特征点字典，若未找到明显盐丘则返回含 NaN 的字典。
    """
    res = {'top_x': np.nan, 'top_y': np.nan, 'base_x': np.nan, 'base_y': np.nan, 'width': np.nan, 'relief': np.nan}
    
    if len(x_salt) < 10:
        return res

    # 1. 盐层剖面提取
    x_min, x_max = np.min(x_salt), np.max(x_salt)
    bins = np.linspace(x_min, x_max, NUM_BINS + 1)
    stat, bin_edges, _ = binned_statistic(x_salt, y_salt, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid = ~np.isnan(stat)
    x_prof, y_prof = bin_centers[valid], stat[valid]
    
    # 2. 平滑与峰值检测
    y_smooth = apply_savgol_filter(y_prof, EXTRACT_SMOOTH_WINDOW)
    peaks, _ = find_peaks(y_smooth, prominence=35.0)
    
    if len(peaks) == 0:
        return res

    # 3. 确定主峰
    peak_idx = peaks[np.argmax(y_smooth[peaks])]
    res['top_x'], res['top_y'] = x_prof[peak_idx], y_smooth[peak_idx]
    
    # 4. 基点追踪 (基于坡度变化或全局极小)
    dy_dx = np.gradient(y_smooth, x_prof)
    abs_slope = np.abs(dy_dx)
    
    # 缺省基点 (左侧端点)
    base_idx = 0
    
    if PUSHING_WALL_SIDE.lower() == 'right':
        # 从峰顶向左扫描，寻找坡度平缓且高度较低的区域作为基底锚点
        depth_limit = np.min(y_smooth[:peak_idx]) + 50.0 # 容差
        scan_range = range(peak_idx - 5, -1, -1)
        for i in scan_range:
            if np.mean(abs_slope[i:i+3]) < FLANK_SLOPE_THRESHOLD and y_smooth[i] <= depth_limit:
                base_idx = i
                break
    
    res['base_x'], res['base_y'] = x_prof[base_idx], y_smooth[base_idx]
    
    # 5. 计算衍生参数
    res['relief'] = res['top_y'] - res['base_y']
    res['width'] = abs(res['top_x'] - res['base_x'])
    
    # 阈值拦截
    if res['relief'] < MIN_RELIEF_THRESHOLD:
        for k in res: res[k] = np.nan
        
    return res

# ==========================================
# 2. 任务分发与主流程
# ==========================================

def process_single_file(dat_path: str, initial_right_wall: Optional[float]) -> Tuple[int, Optional[Dict], Optional[Dict]]:
    """单文件处理单元。"""
    step = extract_step_from_filename(dat_path)
    try:
        df_group, df_coord, right_wall_x = parse_zdem_dat_core(dat_path)
        
        if df_group.empty or df_coord.empty:
            return step, None, None
            
        shortening = (initial_right_wall - right_wall_x) if (right_wall_x and initial_right_wall) else np.nan
        
        # 统一列名处理
        df_coord.columns = [c.lower() for c in df_coord.columns]
        df_group.columns = [c.lower() for c in df_group.columns]
        
        # 转换为数值
        for c in ['id', 'x', 'y']: df_coord[c] = pd.to_numeric(df_coord[c], errors='coerce')
        df_group['id'] = pd.to_numeric(df_group['id'], errors='coerce')
        df_coord.dropna(subset=['id', 'x', 'y'], inplace=True)
        
        # 筛选盐层
        df_merged = pd.merge(df_coord, df_group[['id', 'group']], on='id')
        salt_df = df_merged[df_merged['group'].astype(str).str.lower() == SALT_GROUP_NAME]
        
        kin = detect_salt_kinematics(salt_df['x'].values, salt_df['y'].values) if not salt_df.empty else {}
        
        res_row = {
            'Step': step,
            'Actual_Shortening': shortening,
            'Width': kin.get('width', np.nan),
            'Relief': kin.get('relief', np.nan),
            'Aspect_Ratio': (kin['relief'] / kin['width']) if kin.get('width', 0) > 0 else np.nan
        }
        
        # 缓存剖面数据用于交互修正
        profile_cache = {
            'step': step,
            'x': np.linspace(np.min(df_coord['x']), np.max(df_coord['x']), NUM_BINS), # 简化的 x 轴
            'y': np.nan, # 后面在 01b 实时重算或此处存储
            'top_x': kin.get('top_x', np.nan), 'top_y': kin.get('top_y', np.nan),
            'base_x': kin.get('base_x', np.nan), 'base_y': kin.get('base_y', np.nan)
        }
        # 为保持 01b 兼容，此处需要存储完整的盐层剖面 y
        if not salt_df.empty:
            # 重新分箱获取剖面 y 存储
            xb = np.linspace(np.min(salt_df['x']), np.max(salt_df['x']), NUM_BINS)
            yb, _, _ = binned_statistic(salt_df['x'], salt_df['y'], statistic='max', bins=len(xb))
            profile_cache['x'] = xb
            profile_cache['y'] = yb
            
        return step, res_row, profile_cache

    except Exception as e:
        logging.error(f"处理失败 [{os.path.basename(dat_path)}]: {e}")
        return step, None, None

def main():
    if not os.path.exists(FINAL_OUTPUT_DIR): os.makedirs(FINAL_OUTPUT_DIR)

    for group in EXPERIMENT_GROUPS:
        base_dir = group['base_dir']
        data_dir = os.path.join(base_dir, 'data')
        dat_files = sorted(glob.glob(os.path.join(data_dir, '*.dat')), key=extract_step_from_filename)
        
        if not dat_files: continue
            
        _, _, initial_wall = parse_zdem_dat_core(dat_files[0])
        
        results, profiles = [], {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f, initial_wall) for f in dat_files]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(dat_files), desc=f"Extracting {group['label']}"):
                step, res, prof = f.result()
                if res: results.append(res)
                if prof: profiles[step] = prof

        # 后处理与保存
        df = pd.DataFrame(results).sort_values('Step').reset_index(drop=True)
        df['Shortening_km'] = df['Actual_Shortening'].interpolate() / 1000.0
        
        # 仅保留采样帧（可选，或保留全部供 01b 筛选）
        csv_path = os.path.join(base_dir, CSV_FILENAME)
        pkl_path = os.path.join(base_dir, PKL_FILENAME)
        
        df.to_csv(csv_path, index=False)
        with open(pkl_path, 'wb') as f: pickle.dump(profiles, f)
        
        logging.info(f"组 {group['label']} 提取完成。")

if __name__ == '__main__':
    main()
