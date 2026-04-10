"""
ZDEM Salt Kinematics 跨组联合对比工具 (V2.0)

职责: 整合多个物理实验组的运动学指标，生成学术出版级的联合对比图谱。
工程化改进: 接入 utils 学术样式、自适应多指标渲染、生成矢量 PDF 支持。
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from config import *
from utils import setup_academic_style

# 初始化样式
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
setup_academic_style()

def plot_comparison(metric: str, ylabel: str, filename: str, groups_data: List[Dict]):
    """渲染多组指标对比图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for g in groups_data:
        df = g['df']
        mask_break = df['Extruded_Area'] > 0
        df_pre = df[~mask_break]
        df_post = df[mask_break]
        
        # 绘制实线段 (Pre-extrusion)
        ax.plot(df_pre['Shortening_km'], df_pre[metric], 
                color=g['color'], marker=g['marker'], ms=6, lw=1.5, label=g['label'])
        
        # 绘制虚线段 (Post-extrusion)
        if not df_post.empty:
            last_pre = df_pre.tail(1) if not df_pre.empty else pd.DataFrame()
            df_post_plot = pd.concat([last_pre, df_post])
            ax.plot(df_post_plot['Shortening_km'], df_post_plot[metric], 
                    color=g['color'], linestyle='--', marker=g['marker'], ms=5, lw=1.2)

    ax.set_xlabel("Shortening (km)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, MAX_SHORTENING_KM)
    if 'Aspect_Ratio' in metric: ax.set_ylim(0, MAX_ASPECT_RATIO)
    
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{filename}.pdf"), dpi=300)
    plt.close()

def main():
    if not os.path.exists(FINAL_OUTPUT_DIR): os.makedirs(FINAL_OUTPUT_DIR)
    
    processed_groups = []
    for group in EXPERIMENT_GROUPS:
        csv_path = os.path.join(group['base_dir'], CSV_FILENAME)
        if not os.path.exists(csv_path): continue
        
        df = pd.read_csv(csv_path)
        processed_groups.append({
            'label': group['label'],
            'color': group['color'],
            'marker': group['marker'],
            'df': df
        })
    
    if not processed_groups:
        logging.error("未发现可对比的实验数据。")
        return

    # 导出核心指标
    plot_comparison('Width_Smooth', 'Half-Width (m)', 'Multi_Comparison_Width', processed_groups)
    plot_comparison('Relief_Smooth', 'Relief (m)', 'Multi_Comparison_Relief', processed_groups)
    plot_comparison('Aspect_Ratio_Smooth', 'Aspect Ratio', 'Multi_Comparison_AspectRatio', processed_groups)
    
    logging.info(f"全组对比图已输出至 {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
