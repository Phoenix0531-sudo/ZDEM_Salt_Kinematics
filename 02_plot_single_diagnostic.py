"""
ZDEM Salt Kinematics 诊断图渲染器 (V2.0)

职责: 生成单组实验的运动学演化曲线与盐体剖面形态诊断矩阵。
工程化改进: 接入 utils 学术样式、模块化渲染逻辑、提升出图质量。
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from config import *
from utils import setup_academic_style, apply_savgol_filter

# 配置日志与学术样式
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
setup_academic_style()

def render_diagnostic_plots(group_config: Dict[str, Any]):
    """渲染指定组的诊断图。"""
    base_dir = group_config['base_dir']
    label = group_config['label']
    
    csv_path = os.path.join(base_dir, CSV_FILENAME)
    pkl_path = os.path.join(base_dir, PKL_FILENAME)
    
    if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
        logging.warning(f"数据缺失: {base_dir}")
        return

    df = pd.read_csv(csv_path)
    with open(pkl_path, 'rb') as f:
        profiles = pickle.load(f)

    # 1. 运动学演化轨迹图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 分段绘制（出露前/后）
    mask_break = df['Extruded_Area'] > 0
    df_pre = df[~mask_break]
    df_post = df[mask_break]
    
    color = '#0056b3' # 学术蓝
    if not df_pre.empty:
        ax.plot(df_pre['Shortening_km'], df_pre['Aspect_Ratio_Smooth'], 
                color=color, marker='o', ms=6, label=f"{label} (Pre-extrusion)")
    if not df_post.empty:
        # 衔接点处理
        last_pre = df_pre.tail(1) if not df_pre.empty else pd.DataFrame()
        df_post_plot = pd.concat([last_pre, df_post])
        ax.plot(df_post_plot['Shortening_km'], df_post_plot['Aspect_Ratio_Smooth'], 
                color=color, linestyle='--', marker='s', ms=5, label=f"{label} (Post-extrusion)")

    ax.set_xlabel("Shortening (km)")
    ax.set_ylabel("Aspect Ratio")
    ax.set_xlim(0, MAX_SHORTENING_KM)
    ax.set_ylim(0, MAX_ASPECT_RATIO)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(base_dir, 'Kinematic_Evolution_Diagnostic.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 盐体形态诊断矩阵 (Grid)
    steps = sorted(profiles.keys())
    if not steps: return

    n = len(steps)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, step in enumerate(steps):
        ax = axes[i]
        p = profiles[step]
        x, y = p['x'], p['y']
        y_sm = apply_savgol_filter(y, EXTRACT_SMOOTH_WINDOW)
        
        ax.fill_between(x, np.min(y_sm), y_sm, color='#B2182B', alpha=0.15)
        ax.plot(x, y_sm, color='#B2182B', lw=1.5)
        
        if not np.isnan(p['top_x']):
            ax.scatter(p['top_x'], p['top_y'], marker='*', s=100, color='#B2182B', zorder=5)
        if not np.isnan(p['base_x']):
            ax.scatter(p['base_x'], p['base_y'], marker='o', s=40, color='#4C72B0', zorder=5)

        shortening = df[df['Step'] == step]['Shortening_km'].values[0]
        ax.set_title(f"Step {step} | {shortening:.2f} km", fontsize=10)
        ax.set_xlim(0, MODEL_WIDTH)
        ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
        ax.axis('off')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'Salt_Profiles_Diagnostic_Grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"诊断图已输出至 {base_dir}")

def main():
    for group in EXPERIMENT_GROUPS:
        render_diagnostic_plots(group)

if __name__ == '__main__':
    main()
