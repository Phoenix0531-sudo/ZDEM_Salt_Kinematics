# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false
"""
ZDEM Salt Kinematics 诊断图渲染器 (已还原至原始学术版本)

职责: 生成单组实验的运动学演化曲线与盐体剖面形态诊断矩阵。
工程化改进: 保持模块化结构，还原原始学术绘图风格 (Y 轴右置, 实/虚线逻辑, 特定配色)。
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Any

from config import *
from utils import setup_project_logging, GroupDataManager

# 初始化项目级日志
setup_project_logging()

def render_diagnostic_plots(mgr: GroupDataManager):
    """
    渲染并导出指定实验组的形态学诊断图谱。
    
    还原点:
    - Y 轴右置, 隐藏左/上边框
    - 出露前实线带圆点, 出露后虚线
    - 剖面图使用 lightpink 填充与 crimson 边界
    """
    if not os.path.exists(mgr.csv_path) or not os.path.exists(mgr.pkl_path):
        logging.warning(f"跳过实验组 [{mgr.folder_name}]: 缺少必要的数据文件 (CSV/PKL)。")
        return

    try:
        df_sampled = pd.read_csv(mgr.csv_path)
        with open(mgr.pkl_path, 'rb') as f:
            profiles_data_store = pickle.load(f)
    except Exception as e:
        logging.error(f"加载数据失败 [{mgr.folder_name}]: {e}")
        return

    logging.info(f"正在为实验组 [{mgr.folder_name}] 渲染诊断图...")

    # 配置学术绘图全局参数 (还原至原始 14pt)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5 
    
    # -----------------------------------------------------
    # 图 1：颗粒运动学演化轨迹图 (还原原始视觉逻辑)
    # -----------------------------------------------------
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')  
    ax.grid(False)
    
    # 精简边框 (隐藏左上)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Y 轴右置
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='x', direction='in', top=False, bottom=True, length=6, width=1.5)
    ax.tick_params(axis='y', direction='in', left=False, right=True, length=6, width=1.5)
    
    ax.set_xlim(0, MAX_SHORTENING_KM)
    ax.set_ylim(0, MAX_ASPECT_RATIO)
    ax.set_xlabel('Shortening (km)', weight='bold') 
    ax.set_ylabel('Aspect ratio', weight='bold') 
    
    # 阶段判定逻辑
    breakthrough_steps = np.asarray(df_sampled[df_sampled['Extruded_Area'] > 0]['Step'])
    if len(breakthrough_steps) > 0:
        cutoff_step = breakthrough_steps[0]
        df_pre = df_sampled[df_sampled['Step'] <= cutoff_step]
        df_post = df_sampled[df_sampled['Step'] >= cutoff_step] 
    else:
        df_pre = df_sampled
        df_post = pd.DataFrame(columns=df_sampled.columns)
        
    academic_blue = '#0056b3'
        
    if not df_pre.empty:
        ax.plot(df_pre['Shortening_km'], df_pre['Aspect_Ratio_Smooth'], 
                color=academic_blue, marker='o', markersize=9, 
                markerfacecolor=academic_blue, markeredgecolor=academic_blue,
                linestyle='-', linewidth=2, label=mgr.label)
                
    if not df_post.empty and not df_pre.empty:
        ax.plot(df_post['Shortening_km'], df_post['Aspect_Ratio_Smooth'], 
                color=academic_blue, linestyle='--', linewidth=2)
        
    handles, labels_list = ax.get_legend_handles_labels()
    if not df_post.empty:
        proxy_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Salt extrusion (dashed line)')
        handles.append(proxy_line)
        labels_list.append(proxy_line.get_label())
    
    if handles:
        ax.legend(handles, labels_list, loc='upper left', frameon=False)
    
    ax.margins(x=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(mgr.base_dir, 'Kinematic_Evolution_Diagnostic.png'), dpi=600, bbox_inches='tight')
    plt.close() 

    # -----------------------------------------------------
    # 图 2：盐体剖面形态演化多宫格诊断图 (还原原始配色与标注)
    # -----------------------------------------------------
    valid_sampled_steps = df_sampled['Step'].tolist()
    plot_steps = [s for s in valid_sampled_steps if s in profiles_data_store]
    
    if plot_steps:
        num_plots = len(plot_steps)
        cols = 2
        rows = int(np.ceil(num_plots / cols))
        
        _, axes_prof = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        if num_plots > 1:
            axes_prof = axes_prof.flatten()
        else:
            axes_prof = [axes_prof]
            
        for idx, step in enumerate(plot_steps):
            ax_p = axes_prof[idx]
            p_data = profiles_data_store[step]
            
            # 还原配色: lightpink 填充, crimson 边界线
            ax_p.fill_between(p_data['x'], 0, p_data['y'], color='lightpink', alpha=0.8, label='Salt Body')
            ax_p.plot(p_data['x'], p_data['y'], color='crimson', linewidth=2)
            
            # 还原关键点标注: 红色五角星 (主峰), 蓝色倒三角 (基点)
            ax_p.scatter([p_data['top_x']], [p_data['top_y']], color='red', marker='*', s=200, zorder=5)
            ax_p.scatter([p_data['base_x']], [p_data['base_y']], color='blue', marker='v', s=100, zorder=5)
            
            shortening_arr = np.asarray(df_sampled[df_sampled['Step'] == step]['Shortening_km'])
            shortening = shortening_arr[0] if len(shortening_arr) > 0 else 0.0
            ax_p.set_title(f'Step: {step} | Shortening: {shortening:.1f} km', fontweight='bold', fontsize=12)
            
            ax_p.set_xlim(0, MODEL_WIDTH)
            ax_p.set_ylim(0, MODEL_HEIGHT)
            ax_p.tick_params(labelsize=10)
            
        for i in range(num_plots, len(axes_prof)):
            axes_prof[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(mgr.base_dir, 'Salt_Profiles_Diagnostic_Grid.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logging.info(f"实验组 [{mgr.folder_name}] 诊断图谱渲染完成。")

def main():
    """主程序入口。"""
    for group in EXPERIMENT_GROUPS:
        mgr = GroupDataManager(group)
        render_diagnostic_plots(mgr)

if __name__ == '__main__':
    main()
