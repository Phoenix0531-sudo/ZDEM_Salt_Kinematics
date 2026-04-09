# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false, reportUnknownParameterType=false, reportReturnType=false, reportAny=false
"""
ZDEM 盐构造运动学数据提取器

功能: 批量解析离散元 .dat 输出文件，提取盐丘形态参数
      (起伏度、半宽、高宽比、出露面积) 并序列化至 CSV 与 PKL。
"""
import os
import glob
import re
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import binned_statistic
import concurrent.futures
from tqdm import tqdm

# ==========================================
# 1. 全局配置与日志初始化
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from config import *

# ==========================================
# 2. 核心数据解析函数
# ==========================================
def extract_step(filename: str) -> int:
    """从文件名中提取时间步编号。"""
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0

def parse_zdem_dat(dat_path: str):
    """
    解析 ZDEM .dat 格式的颗粒数据文件。

    返回:
        df_group: 颗粒-组映射 DataFrame
        df_coord: 颗粒坐标 DataFrame
        right_wall_x: 右侧墙体 X 坐标 (若存在)
    """
    with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    group_data, coord_data = [], []
    group_cols, coord_cols = [], []
    in_group, in_coord, in_wall = False, False, False
    right_wall_x = None

    for line in lines:
        line_lower = line.lower()
        if 'p1[0]' in line_lower and 'p2[0]' in line_lower:
            in_wall = True
            continue
            
        if in_wall:
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                in_wall = False
            else:
                if len(parts) >= 6:
                    wall_id = parts[1]
                    if wall_id == '2':  
                        right_wall_x = float(parts[2])  
                        in_wall = False 
        
        parts = line.strip().split()
        if not parts:
            in_group, in_coord = False, False
            continue

        if 'group' in line_lower and 'fric' in line_lower:
            in_group, in_coord = True, False
            group_cols = line_lower.split()
            continue
            
        elif 'rad' in line_lower and 'color' in line_lower and 'x' in line_lower:
            in_coord, in_group = True, False
            coord_cols = line_lower.split()
            continue

        is_data_line = parts[0].replace('.', '', 1).replace('-', '', 1).isdigit()
        if is_data_line:
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

def process_single_file(dat_path: str, initial_right_wall: float | None):
    """
    处理单个 .dat 文件，提取盐丘形态学参数。

    算法流程:
        1. 解析颗粒坐标与组信息
        2. 基于分箱统计构建全局/盐层地表剖面
        3. 基于一阶导数与动态深度约束追踪非挤压端形变前锋
        4. 计算半宽、起伏度与高宽比
    """
    step = extract_step(dat_path)
    df_group, df_coord, right_wall_x = parse_zdem_dat(dat_path)
    
    result_dict = None
    profile_dict = None
    
    if df_group.empty or df_coord.empty:
        return step, result_dict, profile_dict
        
    actual_shortening = np.nan
    if right_wall_x is not None and initial_right_wall is not None:
        actual_shortening = initial_right_wall - right_wall_x
        
    df_coord.rename(columns=lambda x: str(x).lower(), inplace=True)
    df_group.rename(columns=lambda x: str(x).lower(), inplace=True)
    
    df_coord['id'] = pd.to_numeric(df_coord['id'], errors='coerce')
    df_coord['x'] = pd.to_numeric(df_coord['x'], errors='coerce')
    df_coord['y'] = pd.to_numeric(df_coord['y'], errors='coerce')
    df_coord = df_coord.dropna(subset=['id', 'x', 'y'])
    
    df_group['id'] = pd.to_numeric(df_group['id'], errors='coerce')
    df_group = df_group.dropna(subset=['id'])
    
    x_all = np.asarray(df_coord['x'], dtype=float)
    y_all = np.asarray(df_coord['y'], dtype=float)
    if len(x_all) == 0: 
        return step, result_dict, profile_dict

    x_min, x_max = np.min(x_all), np.max(x_all)
    bins = np.linspace(x_min, x_max, NUM_BINS + 1)
    
    # 全局地表剖面：按分箱取纵坐标极大值
    stat, bin_edges, _ = binned_statistic(x_all, y_all, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    valid_mask = ~np.isnan(stat)
    x_surface = bin_centers[valid_mask]
    y_surface = stat[valid_mask]
    
    # 合并颗粒坐标与组信息，筛选盐层颗粒
    df_merged = pd.merge(df_coord, df_group[['id', 'group']], on='id', how='inner')
    salt_df = df_merged[df_merged['group'].astype(str).str.lower() == SALT_GROUP_NAME]
    
    extruded_area = 0.0
    dynamic_aspect_ratio = np.nan
    temp_width = np.nan
    temp_relief = np.nan
    
    if not salt_df.empty:
        salt_x_coords = np.asarray(salt_df['x'], dtype=float)
        salt_y_coords = np.asarray(salt_df['y'], dtype=float)
        
        # 基于全局地表插值判定盐层颗粒出露状态
        global_surface_y_for_salt = np.interp(salt_x_coords, x_surface, y_surface)
        extruded_mask = salt_y_coords >= (global_surface_y_for_salt - 1.5 * PARTICLE_RADIUS)
        extruded_area = np.sum(extruded_mask) * PARTICLE_AREA
        
        try:
            # 构建盐层顶面剖面
            salt_stat, salt_bin_edges, _ = binned_statistic(salt_x_coords, salt_y_coords, statistic='max', bins=bins)
            salt_bin_centers = (salt_bin_edges[:-1] + salt_bin_edges[1:]) / 2.0
            salt_valid_mask = ~np.isnan(salt_stat)
            x_salt_surf = salt_bin_centers[salt_valid_mask]
            y_salt_surf = salt_stat[salt_valid_mask]
            
            if len(x_salt_surf) > 10:
                # Savitzky-Golay 平滑盐层顶面剖面
                window_len = min(EXTRACT_SMOOTH_WINDOW, len(y_salt_surf))
                if window_len % 2 == 0: window_len -= 1
                y_smooth = np.asarray(savgol_filter(y_salt_surf, window_length=window_len, polyorder=3), dtype=float)
                
                # 识别盐丘主峰
                peaks, _ = find_peaks(y_smooth, prominence=35.0)
                peaks = np.asarray(peaks, dtype=int)
                
                if len(peaks) == 0:
                    fallback_peak_idx = np.argmax(y_smooth)
                    if fallback_peak_idx < 10 or fallback_peak_idx > len(y_smooth) - 10:
                         fallback_peak_idx = len(y_smooth) // 2
                    peaks = np.array([fallback_peak_idx])
                
                if len(peaks) > 0:
                    central_peak_idx = peaks[np.argmax(y_smooth[peaks])]
                    central_peak_y = y_smooth[central_peak_idx]
                    top_x = x_salt_surf[central_peak_idx]
                    top_y = central_peak_y
                    
                    left_profile_y = y_smooth[:central_peak_idx]
                    right_profile_y = y_smooth[central_peak_idx:]
                    
                    temp_width = 0.0
                    temp_relief = 0.0
                    
                    if len(left_profile_y) > 0 and len(right_profile_y) > 0:
                        # ==========================================
                        # 向非挤压端形变前锋追踪 (Outward Deformation Front Tracking)
                        # 基于一阶导数与动态深度约束，从核部向非挤压端扫描真实形变前锋
                        # ==========================================
                        dy_dx = np.gradient(y_smooth, x_salt_surf)
                        abs_slope = np.abs(dy_dx)

                        base_x, base_y = x_salt_surf[0], y_smooth[0]

                        if PUSHING_WALL_SIDE.lower() == 'right':
                            # 挤压端在右侧 → 向左侧（非挤压端）追踪形变前锋
                            left_profile = y_smooth[:central_peak_idx]
                            
                            # 非挤压端区域最低盆地海拔
                            global_min_y = np.min(left_profile)
                            
                            # 动态深度容差：允许高出最低点 5% 的起伏，下限 50 m
                            depth_tolerance = global_min_y + max(50.0, 0.05 * (central_peak_y - global_min_y))
                            
                            SCAN_WINDOW = 5
                            left_min_idx = 0  # 缺省回退至剖面起点
                            
                            # 从主峰附近向左逆向扫描，双重物理约束锚定基点
                            for i in range(central_peak_idx - SCAN_WINDOW, -1, -1):
                                local_avg_slope = np.mean(abs_slope[i : i + SCAN_WINDOW])
                                
                                # 斜率低于阈值 且 海拔低于深度容差 → 锚定为形变前锋
                                if local_avg_slope < FLANK_SLOPE_THRESHOLD and y_smooth[i] <= depth_tolerance:
                                    left_min_idx = i
                                    break
                            
                            base_x = x_salt_surf[left_min_idx]
                            base_y = y_smooth[left_min_idx]

                        # 基于单侧锚定点计算半宽与起伏度
                        temp_width = abs(top_x - base_x)
                        temp_relief = top_y - base_y
                        
                        profile_dict = {
                            'step': step,
                            'x': x_salt_surf, 'y': y_smooth,
                            'top_x': top_x, 'top_y': top_y,
                            'base_x': base_x, 'base_y': base_y
                        }
                    else:
                        base_y = np.min(y_smooth)
                        temp_width = 0.0
                        temp_relief = central_peak_y - base_y
                        
                    # 起伏度低于阈值的帧标记为 NaN，绘图时自动断开曲线
                    if temp_relief < MIN_RELIEF_THRESHOLD:
                        temp_relief = np.nan
                        temp_width = np.nan
                        dynamic_aspect_ratio = np.nan
                    else:
                        if temp_width > 0:
                            dynamic_aspect_ratio = temp_relief / temp_width
                        else:
                            dynamic_aspect_ratio = np.nan
                        
        except Exception as e:
            logging.warning(f"形态参数提取失败 (step={step}): {e}")
            temp_width = np.nan
            temp_relief = np.nan
        
    result_dict = {
        'Step': step,
        'Actual_Shortening': actual_shortening, 
        'Extruded_Area': extruded_area,
        'Width': temp_width,
        'Relief': temp_relief,
        'Aspect_Ratio': dynamic_aspect_ratio
    }
    
    return step, result_dict, profile_dict

# ==========================================
# 3. 主提取流程（外层串行实验组，内层并行文件）
# ==========================================
def main():
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    for group in EXPERIMENT_GROUPS:
        base_dir = group['base_dir']
        
        if not os.path.exists(base_dir):
            logging.error(f"目录缺失: '{base_dir}'，跳过该组。")
            continue
            
        data_dir = os.path.join(base_dir, 'data')
        if not os.path.exists(data_dir):
            logging.error(f"数据子目录缺失: '{data_dir}'，跳过该组。")
            continue
            
        dat_pattern = os.path.join(data_dir, '*.dat')
        dat_files = glob.glob(dat_pattern)
        
        if not dat_files:
            logging.warning(f"未发现 .dat 文件: {data_dir}，跳过。")
            continue
            
        dat_files.sort(key=extract_step)
        _, _, initial_right_wall = parse_zdem_dat(dat_files[0])
        
        results = []
        profiles_data_store = {} 
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = executor.map(process_single_file, dat_files, [initial_right_wall] * len(dat_files))
            # tqdm 进度条美化：ncols 固定宽度，ascii=True，bar_format 统一
            for step, res_dict, prof_dict in tqdm(
                futures,
                total=len(dat_files),
                desc=f"Extraction [{os.path.basename(group['base_dir'])}]",
                unit="file",
                colour="green",
                ncols=80,
                ascii=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            ):
                if res_dict:
                    results.append(res_dict)
                if prof_dict:
                    profiles_data_store[step] = prof_dict

        df = pd.DataFrame(results)
        if df.empty:
            continue
            
        df = df.sort_values(by='Step').reset_index(drop=True)
        # 线性插值填补缺失的墙体位移帧
        df['Actual_Shortening'] = df['Actual_Shortening'].interpolate(method='linear', limit_direction='both')
        # 仅丢弃插值后仍缺失挤压量的无效行，保留 Aspect_Ratio 的 NaN
        df = df.dropna(subset=['Actual_Shortening']).reset_index(drop=True)
        
        df['Shortening_km'] = df['Actual_Shortening'] / 1000.0
        df['Aspect_Ratio_Smooth'] = df['Aspect_Ratio'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        
        if 'Width' in df.columns and 'Relief' in df.columns:
            df['Width_Smooth'] = df['Width'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
            df['Relief_Smooth'] = df['Relief'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        
        # ==========================================
        # 地质演化分段采样 (出露前/后阶段拆分)
        # ==========================================
        sampled_indices = []
        
        # 定位出露临界帧
        true_breakthrough = df[df['Extruded_Area'] > 0]
        
        if not true_breakthrough.empty:
            break_idx = true_breakthrough.index[0]
            critical_shortening = df.loc[break_idx, 'Shortening_km']
            
            # 出露前等距采样
            df_pre_all = df[df.index <= break_idx]
            if not df_pre_all.empty:
                min_s = df_pre_all['Shortening_km'].min()
                target_shortenings = np.linspace(min_s, critical_shortening, PRE_EXTRUSION_FRAMES)
                for ts in target_shortenings:
                    idx = (np.abs(df_pre_all['Shortening_km'] - ts)).idxmin()
                    if idx not in sampled_indices:
                        sampled_indices.append(idx)
            
            # 确保临界帧被纳入
            if break_idx not in sampled_indices:
                sampled_indices.append(break_idx)
            
            # 出露后保留终态帧
            df_post_all = df[df.index > break_idx]
            if not df_post_all.empty and POST_EXTRUSION_FRAMES > 0:
                post_tail = df_post_all.tail(POST_EXTRUSION_FRAMES)
                for pidx in post_tail.index:
                    if pidx not in sampled_indices:
                        sampled_indices.append(pidx)
        else:
            # 全程未出露：全区间等距采样
            min_s = df['Shortening_km'].min()
            max_s = df['Shortening_km'].max()
            target_shortenings = np.linspace(min_s, max_s, PRE_EXTRUSION_FRAMES)
            for ts in target_shortenings:
                idx = (np.abs(df['Shortening_km'] - ts)).idxmin()
                if idx not in sampled_indices:
                    sampled_indices.append(idx)
        
        sampled_indices.sort()
        
        df_sampled = df.loc[sampled_indices].copy()
        valid_sampled_steps = set(df_sampled['Step'].tolist())
        filtered_profiles = {k: v for k, v in profiles_data_store.items() if k in valid_sampled_steps}
        
        csv_path = os.path.join(base_dir, CSV_FILENAME)
        pkl_path = os.path.join(base_dir, PKL_FILENAME)
        
        df_sampled.to_csv(csv_path, index=False)
        with open(pkl_path, 'wb') as f:
            pickle.dump(filtered_profiles, f)

if __name__ == '__main__':
    main()