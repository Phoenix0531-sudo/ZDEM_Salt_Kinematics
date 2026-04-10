# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false, reportUnknownParameterType=false, reportAny=false, reportExplicitAny=false
"""
ZDEM Salt Kinematics 跨组联合对比工具 (已还原至原始学术版本)

职责: 整合多个物理实验组的颗粒运动学指标，生成学术出版级的对比图谱。
还原说明: 
- Y 轴右置 (tick_right)
- 阶段化线型 (出露前带 Marker 实线, 出露后无 Marker 虚线)
- 代理图例逻辑
- 字体大小 14pt, 600 DPI 导出
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Any

from config import *
from utils import setup_project_logging

# 初始化项目级日志
setup_project_logging()

# ==========================================
# 1. 跨组绘图引擎 (还原至原始逻辑)
# ==========================================
def plot_evolution_metric(metric_col: str, y_label: str, file_prefix: str, df_all_groups: list[dict[str, Any]], ylim_max: float | None = None) -> None:
    """
    渲染多组颗粒运动学指标对比图。
    """
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')  
    ax.grid(False)
    
    # 还原边框样式 (隐藏左上)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # 还原 Y 轴右置
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='x', direction='in', top=False, bottom=True, length=6, width=1.5)
    ax.tick_params(axis='y', direction='in', left=False, right=True, length=6, width=1.5)
    
    ax.set_xlim(0, MAX_SHORTENING_KM)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    ax.margins(x=0.15)
        
    ax.set_ylabel(y_label, weight='bold')
    ax.set_xlabel('Shortening (km)', weight='bold')

    has_dashed_post = False
    
    for group_data in df_all_groups:
        color = group_data['color']
        marker = group_data['marker']
        label = group_data['label']
        df_pre = group_data['df_pre']
        df_post = group_data['df_post']
        
        # 阶段一：出露前 (实线带 Marker)
        if not df_pre.empty and metric_col in df_pre.columns:
            ax.plot(df_pre['Shortening_km'], df_pre[metric_col], 
                    color=color, marker=marker, markersize=8, 
                    markerfacecolor=color, markeredgecolor=color, 
                    linestyle='-', linewidth=2, label=label)
            
        # 阶段二：出露后 (虚线无 Marker)
        if not df_post.empty and metric_col in df_post.columns:
            has_dashed_post = True
            ax.plot(df_post['Shortening_km'], df_post[metric_col], 
                    color=color, marker='', linestyle='--', linewidth=2)

    handles, labels = ax.get_legend_handles_labels()
    
    # 注入代理图例
    if has_dashed_post:
        proxy_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Salt extrusion (dashed lines)')
        handles.append(proxy_line)
        labels.append(proxy_line.get_label())
    
    ax.legend(handles, labels, loc='upper left', frameon=False)
    
    plt.tight_layout()
    
    plot_png = os.path.join(FINAL_OUTPUT_DIR, f'{file_prefix}.png')
    plot_pdf = os.path.join(FINAL_OUTPUT_DIR, f'{file_prefix}.pdf')
    
    plt.savefig(plot_png, dpi=600, bbox_inches='tight')
    plt.savefig(plot_pdf, dpi=600, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Generated comparison plot: {file_prefix}")


def main():
    """主程序。"""
    if not os.path.exists(FINAL_OUTPUT_DIR): 
        os.makedirs(FINAL_OUTPUT_DIR)
        
    # 严格遵循原始样式配置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5 
    
    df_all_groups = []
    
    for group in EXPERIMENT_GROUPS:
        base_dir = group['base_dir']
        logging.info(f"Loading data: {group['label']}")
        
        if not os.path.exists(base_dir):
            continue
            
        csv_path = os.path.join(base_dir, CSV_FILENAME)
        if not os.path.exists(csv_path):
            continue
            
        try:
            df_sampled = pd.read_csv(csv_path)
            if df_sampled.empty:
                continue
            
            # 还原阶段拆分逻辑
            breakthrough_df = df_sampled[df_sampled['Extruded_Area'] > 0]
            if not breakthrough_df.empty:
                cutoff_step = np.asarray(breakthrough_df['Step'])[0]
                df_pre = df_sampled[df_sampled['Step'] <= cutoff_step]
                df_post = df_sampled[df_sampled['Step'] >= cutoff_step] 
            else:
                df_pre = df_sampled
                df_post = pd.DataFrame(columns=df_sampled.columns)
                
            df_all_groups.append({
                'color': group['color'],
                'marker': group['marker'],
                'label': group['label'],
                'df_pre': df_pre,
                'df_post': df_post
            })
        except Exception:
            continue

    # 还原指标绘图顺序
    plot_evolution_metric('Width_Smooth', 'Half-Width (m)', 'Multi_Evolution_HalfWidth', df_all_groups)
    plot_evolution_metric('Relief_Smooth', 'Relief (m)', 'Multi_Evolution_Relief', df_all_groups)
    plot_evolution_metric('Aspect_Ratio_Smooth', 'Aspect ratio', 'Multi_Evolution_AspectRatio', df_all_groups, ylim_max=MAX_ASPECT_RATIO)

    logging.info(f"All comparison plots restored in {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
