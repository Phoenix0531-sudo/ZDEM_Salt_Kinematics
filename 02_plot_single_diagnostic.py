"""
ZDEM Salt Kinematics 诊断图渲染器

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
from utils import (
    setup_academic_style, 
    apply_savgol_filter, 
    GroupDataManager,
    setup_project_logging
)

setup_project_logging()
setup_academic_style()

def render_diagnostic_plots(mgr: GroupDataManager):
    """
    渲染并导出指定实验组的形态学诊断图谱。
    
    包含：
    1. 颗粒运动学演化轨迹图（宽高比 vs 缩短量）。
    2. 盐体剖面形态演化矩阵（多宫格 Grid）。
    """
    if not os.path.exists(mgr.csv_path) or not os.path.exists(mgr.pkl_path):
        logging.warning(f"缺少必要依赖文件，跳过组别 [{mgr.label}]。")
        return

    try:
        df = pd.read_csv(mgr.csv_path)
        with open(mgr.pkl_path, 'rb') as f:
            profiles = pickle.load(f)
    except Exception as e:
        logging.error(f"读取数据失败 [{mgr.label}]: {e}")
        return

    # 1. 颗粒运动学演化轨迹图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 区分盐体颗粒出露地表前后的演化阶段
    mask_break = df['Extruded_Area'] > 0
    df_pre = df[~mask_break]
    df_post = df[mask_break]
    
    color = '#0056b3'
    if not df_pre.empty:
        ax.plot(df_pre['Shortening_km'], df_pre['Aspect_Ratio_Smooth'], 
                color=color, marker='o', ms=6, label=f"{mgr.label} (出露前)")
    if not df_post.empty:
        # 衔接出露前后的曲线
        last_pre = df_pre.tail(1) if not df_pre.empty else pd.DataFrame()
        df_post_plot = pd.concat([last_pre, df_post])
        ax.plot(df_post_plot['Shortening_km'], df_post_plot['Aspect_Ratio_Smooth'], 
                color=color, linestyle='--', marker='s', ms=5, label=f"{mgr.label} (出露后)")

    ax.set_xlabel("构造缩短量 (km)")
    ax.set_ylabel("盐体宽高比 (Aspect Ratio)")
    ax.set_xlim(0, MAX_SHORTENING_KM)
    ax.set_ylim(0, MAX_ASPECT_RATIO)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    out_path = os.path.join(mgr.base_dir, 'Kinematic_Evolution_Diagnostic.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 颗粒形态诊断矩阵 (Grid)
    steps = sorted(profiles.keys())
    if not steps: 
        return

    # 选取关键步进行展示（最多 15 个）
    display_steps = steps if len(steps) <= 15 else [steps[i] for i in np.linspace(0, len(steps)-1, 15, dtype=int)]
    n = len(display_steps)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, step in enumerate(display_steps):
        ax = axes[i]
        p = profiles[step]
        x, y = p['x'], p['y']
        
        if len(x) == 0:
            ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"Step {step}", fontsize=10)
            ax.axis('off')
            continue

        # 平滑处理颗粒包络线
        y_sm = apply_savgol_filter(y, EXTRACT_SMOOTH_WINDOW)
        ax.fill_between(x, np.min(y_sm), y_sm, color='#B2182B', alpha=0.15)
        ax.plot(x, y_sm, color='#B2182B', lw=1.5, label='盐体包络线')
        
        # 标注颗粒运动学关键特征点
        if not np.isnan(p['top_x']):
            ax.scatter(p['top_x'], p['top_y'], marker='*', s=100, color='#B2182B', zorder=5, label='盐体主峰')
        if not np.isnan(p['base_x']):
            ax.scatter(p['base_x'], p['base_y'], marker='o', s=40, color='#4C72B0', zorder=5, label='识别基点')

        shortening = df[df['Step'] == step]['Shortening_km'].values[0]
        ax.set_title(f"Step {step} | 缩短量: {shortening:.2f} km", fontsize=10)
        ax.set_xlim(0, MODEL_WIDTH)
        ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
        ax.axis('off')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)): 
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(mgr.base_dir, 'Salt_Profiles_Diagnostic_Grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"组别 [{mgr.label}] 颗粒运动学诊断图谱渲染完成。")

def main():
    for group in EXPERIMENT_GROUPS:
        mgr = GroupDataManager(group)
        render_diagnostic_plots(mgr)

if __name__ == '__main__':
    main()
