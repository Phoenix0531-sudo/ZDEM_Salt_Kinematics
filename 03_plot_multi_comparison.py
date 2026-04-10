"""
ZDEM Salt Kinematics 跨组联合对比工具

职责: 整合多个物理实验组的颗粒运动学指标，生成学术出版级的对比图谱。
工程化改进: 接入统一日志、术语规范化、支持矢量 PDF 导出、优化图例布局。
"""
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from config import *
from utils import setup_academic_style, setup_project_logging, GroupDataManager

# 初始化工程化组件
setup_project_logging()
setup_academic_style()

def plot_comparison(metric: str, ylabel: str, filename: str, groups_data: list[dict[str, Any]]):
    """
    渲染多组颗粒运动学指标对比图。
    
    Parameters
    ----------
    metric : str
        DataFrame 中的指标列名。
    ylabel : str
        纵坐标显示的学术标签 (支持 LaTeX)。
    filename : str
        导出的文件名（不含扩展名）。
    groups_data : list[dict]
        包含各实验组 df, label, color, marker 的列表。
    """
    _, ax = plt.subplots(figsize=(8, 6))
    
    for g in groups_data:
        df = g['df']
        if metric not in df.columns:
            continue
            
        # 识别颗粒出露前后的演化阶段（通过 Extruded_Area 判定）
        mask_break = df['Extruded_Area'] > 0
        df_pre = df[~mask_break]
        df_post = df[mask_break]
        
        # 阶段一：出露前 (实线渲染)
        ax.plot(df_pre['Shortening_km'], df_pre[metric], 
                color=g['color'], marker=g['marker'], ms=6, lw=1.5, 
                label=f"{g['label']} (出露前)")
        
        # 阶段二：出露后 (虚线渲染，区分演化阶段)
        if not df_post.empty:
            # 衔接点平滑处理
            last_pre = df_pre.tail(1) if not df_pre.empty else pd.DataFrame()
            df_post_plot = pd.concat([last_pre, df_post])
            ax.plot(df_post_plot['Shortening_km'], df_post_plot[metric], 
                    color=g['color'], linestyle='--', marker=g['marker'], ms=5, lw=1.2,
                    label=f"{g['label']} (出露后)")

    ax.set_xlabel("构造缩短量 (Shortening, km)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, MAX_SHORTENING_KM)
    
    # 针对特定指标设置纵轴显示策略
    if 'Aspect_Ratio' in metric: 
        ax.set_ylim(0, MAX_ASPECT_RATIO)
    elif 'Relief' in metric:
        ax.set_ylim(0, None) 
        
    # 优化图例：去重并美化布局
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=10, loc='best')
    
    # 移除冗余边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    
    # 同时输出位图与矢量图，满足不同出版需求
    png_path = os.path.join(FINAL_OUTPUT_DIR, f"{filename}.png")
    pdf_path = os.path.join(FINAL_OUTPUT_DIR, f"{filename}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主程序入口：加载数据并生成盐底辟演化图。"""
    if not os.path.exists(FINAL_OUTPUT_DIR): 
        os.makedirs(FINAL_OUTPUT_DIR)
    
    processed_groups: list[dict[str, Any]] = []
    for group in EXPERIMENT_GROUPS:
        mgr = GroupDataManager(group)
        if not os.path.exists(mgr.csv_path): 
            logging.warning(f"跳过实验组 [{mgr.folder_name}]: 缺少提取的 CSV 指标文件。")
            continue
        
        try:
            df = pd.read_csv(mgr.csv_path)
            processed_groups.append({
                'label': mgr.label,
                'color': group['color'],
                'marker': group['marker'],
                'df': df
            })
        except Exception as e:
            logging.error(f"加载组别数据失败 [{mgr.folder_name}]: {e}")
    
    if not processed_groups:
        logging.error("未检测到可用的实验组数据，无法生成对比图。")
        return

    # 渲染盐底辟形态演化图的核心颗粒指标
    logging.info("正在生成盐底辟形态演化图...")
    
    plot_comparison('Width_Smooth', '盐体颗粒半宽 (Half-Width, m)', 'Evolution_Width', processed_groups)
    plot_comparison('Relief_Smooth', '盐体颗粒地形起伏 (Relief, m)', 'Evolution_Relief', processed_groups)
    plot_comparison('Aspect_Ratio_Smooth', '盐体宽高比 (Aspect Ratio)', 'Evolution_AspectRatio', processed_groups)
    plot_comparison('Extruded_Area', '盐体颗粒出露面积 (Extruded Area, $m^2$)', 'Evolution_Area', processed_groups)
    
    logging.info(f"所有盐底辟形态演化图已导出至: {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
