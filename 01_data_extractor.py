# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false
import os
import glob
import re
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import binned_statistic
import concurrent.futures
from tqdm import tqdm

# ==========================================
# 1. 全局配置与物理参数
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

FINAL_OUTPUT_DIR = r"E:\0.Information\4.Temp\StructLab\盐构造部分\实验\68"
EXPERIMENT_GROUPS = [
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '150'), 'label': r'$v_e = 150 \ m \cdot s^{-1}$', 'color': 'b', 'marker': 'o'},
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '300'), 'label': r'$v_e = 300 \ m \cdot s^{-1}$', 'color': 'r', 'marker': 's'},
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '600'), 'label': r'$v_e = 600 \ m \cdot s^{-1}$', 'color': 'g', 'marker': '^'}
]

SALT_GROUP_NAME = 'salt'
PARTICLE_RADIUS = 70.0         
PARTICLE_AREA = np.pi * (PARTICLE_RADIUS ** 2)
NUM_BINS = 200                 
SMOOTHING_WINDOW = 51

NUM_KEY_STAGES = 10
MAX_SHORTENING_KM = 24.0

# ==========================================
# 2. 核心数据解析算法
# ==========================================
def extract_step(filename: str) -> int:
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0

def parse_zdem_dat(dat_path: str):
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
    
    stat, bin_edges, _ = binned_statistic(x_all, y_all, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    valid_mask = ~np.isnan(stat)
    x_surface = bin_centers[valid_mask]
    y_surface = stat[valid_mask]
    
    df_merged = pd.merge(df_coord, df_group[['id', 'group']], on='id', how='inner')
    salt_df = df_merged[df_merged['group'].astype(str).str.lower() == SALT_GROUP_NAME]
    
    extruded_area = 0.0
    dynamic_aspect_ratio = np.nan
    
    if not salt_df.empty:
        salt_x_coords = np.asarray(salt_df['x'], dtype=float)
        salt_y_coords = np.asarray(salt_df['y'], dtype=float)
        
        global_surface_y_for_salt = np.interp(salt_x_coords, x_surface, y_surface)
        extruded_mask = salt_y_coords >= (global_surface_y_for_salt - 1.5 * PARTICLE_RADIUS)
        extruded_area = np.sum(extruded_mask) * PARTICLE_AREA
        
        try:
            salt_stat, salt_bin_edges, _ = binned_statistic(salt_x_coords, salt_y_coords, statistic='max', bins=bins)
            salt_bin_centers = (salt_bin_edges[:-1] + salt_bin_edges[1:]) / 2.0
            salt_valid_mask = ~np.isnan(salt_stat)
            x_salt_surf = salt_bin_centers[salt_valid_mask]
            y_salt_surf = salt_stat[salt_valid_mask]
            
            if len(x_salt_surf) > 10:
                window_len = min(21, len(y_salt_surf))
                if window_len % 2 == 0: window_len -= 1
                y_smooth = np.asarray(savgol_filter(y_salt_surf, window_length=window_len, polyorder=3), dtype=float)
                
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
                    
                    if len(left_profile_y) > 0 and len(right_profile_y) > 0:
                        left_min_idx = np.argmin(left_profile_y)
                        right_min_idx = central_peak_idx + np.argmin(right_profile_y)
                        
                        left_base_x = x_salt_surf[left_min_idx]
                        left_base_y = y_smooth[left_min_idx]
                        right_base_x = x_salt_surf[right_min_idx]
                        right_base_y = y_smooth[right_min_idx]
                        
                        y_baseline = (left_base_y + right_base_y) / 2.0
                        temp_relief = central_peak_y - y_baseline
                        
                        profile_dict = {
                            'step': step,
                            'x': x_salt_surf, 'y': y_smooth,
                            'top_x': top_x, 'top_y': top_y,
                            'l_base_x': left_base_x, 'l_base_y': left_base_y,
                            'r_base_x': right_base_x, 'r_base_y': right_base_y,
                            'baseline': y_baseline
                        }
                    else:
                        y_baseline = np.min(y_smooth)
                        temp_relief = central_peak_y - y_baseline
                    
                    widths, width_heights, left_ips, right_ips = peak_widths(y_smooth, [central_peak_idx], rel_height=0.85)
                    idx_left = left_ips[0]
                    idx_right = right_ips[0]
                    x_left = np.interp(idx_left, np.arange(len(x_salt_surf)), x_salt_surf)
                    x_right = np.interp(idx_right, np.arange(len(x_salt_surf)), x_salt_surf)
                    
                    temp_width = x_right - x_left
                    if temp_width > 0:
                        dynamic_aspect_ratio = temp_relief / temp_width
                        
        except Exception as e:
            logging.warning(f"Failed to extract parameter profile for step {step}: {e}")
        
    result_dict = {
        'Step': step,
        'Actual_Shortening': actual_shortening, 
        'Extruded_Area': extruded_area,
        'Aspect_Ratio': dynamic_aspect_ratio
    }
    
    return step, result_dict, profile_dict

# ==========================================
# 3. 主提取循环 (外层串行组，内层并行文件)
# ==========================================
def main():
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    for group in EXPERIMENT_GROUPS:
        base_dir = group['base_dir']
        
        if not os.path.exists(base_dir):
            logging.error(f"Directory missing: The base path '{base_dir}' does not exist. Skipping group.")
            continue
            
        data_dir = os.path.join(base_dir, 'data')
        if not os.path.exists(data_dir):
            logging.error(f"Directory missing: the data subdirectory '{data_dir}' does not exist. Skipping group.")
            continue
            
        dat_pattern = os.path.join(data_dir, '*.dat')
        dat_files = glob.glob(dat_pattern)
        
        if not dat_files:
            logging.warning(f"No .dat files found in {data_dir}. Skipping...")
            continue
            
        dat_files.sort(key=extract_step)
        _, _, initial_right_wall = parse_zdem_dat(dat_files[0])
        
        results = []
        profiles_data_store = {} 
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = executor.map(process_single_file, dat_files, [initial_right_wall] * len(dat_files))
            for step, res_dict, prof_dict in tqdm(futures, total=len(dat_files), desc=f"Extraction [{os.path.basename(group['base_dir'])}]", unit="file", colour="green"):
                if res_dict:
                    results.append(res_dict)
                if prof_dict:
                    profiles_data_store[step] = prof_dict

        df = pd.DataFrame(results)
        if df.empty:
            continue
            
        df = df.sort_values(by='Step').reset_index(drop=True)
        df = df.dropna(subset=['Actual_Shortening', 'Aspect_Ratio']) 
        
        df['Shortening_km'] = df['Actual_Shortening'] / 1000.0
        df['Aspect_Ratio_Smooth'] = df['Aspect_Ratio'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        
        # 丢弃空值后立即重置索引，防止错位
        df = df.dropna(subset=['Aspect_Ratio_Smooth']).reset_index(drop=True)
        
        min_shortening = df['Shortening_km'].min()
        max_shortening = df['Shortening_km'].max()
        target_shortenings = np.linspace(min_shortening, max_shortening, NUM_KEY_STAGES)
        
        sampled_indices = []
        for ts in target_shortenings:
            # 将 .argmin() 改为 .idxmin()，获取真实的 Pandas 行索引
            idx = (np.abs(df['Shortening_km'] - ts)).idxmin()
            if idx not in sampled_indices:
                sampled_indices.append(idx)
                
        # 强制采样修正逻辑：抓取准确的刺穿时刻
        true_breakthrough = df[df['Extruded_Area'] > 0]
        if not true_breakthrough.empty:
            exact_break_idx = true_breakthrough.index[0] 
            if exact_break_idx not in sampled_indices:
                sampled_indices.append(exact_break_idx) 
                sampled_indices.sort() 
                
        df_sampled = df.loc[sampled_indices].copy()
        valid_sampled_steps = set(df_sampled['Step'].tolist())
        filtered_profiles = {k: v for k, v in profiles_data_store.items() if k in valid_sampled_steps}
        
        csv_path = os.path.join(base_dir, 'kinematics_data.csv')
        pkl_path = os.path.join(base_dir, 'profiles_cache.pkl')
        
        df_sampled.to_csv(csv_path, index=False)
        with open(pkl_path, 'wb') as f:
            pickle.dump(filtered_profiles, f)

if __name__ == '__main__':
    main()