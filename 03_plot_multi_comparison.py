# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. 全局配置与渲染器状态
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
MAX_SHORTENING_KM = 24.0

# ==========================================
# 2. 综合多层级绘图核心
# ==========================================
def main():
    if not os.path.exists(FINAL_OUTPUT_DIR): 
        os.makedirs(FINAL_OUTPUT_DIR)
        
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5 
    
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')  
    ax.grid(False)
    
    # 学术级坐标流氓边框剔除
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='x', direction='in', top=False, bottom=True, length=6, width=1.5)
    ax.tick_params(axis='y', direction='in', left=False, right=True, length=6, width=1.5)
    
    _ = ax.set_xlim(0, MAX_SHORTENING_KM)
    _ = ax.set_ylim(0, 0.30)
    _ = ax.set_xlabel('Shortening (km)', weight='bold') 
    _ = ax.set_ylabel('Aspect ratio', weight='bold') 

    has_dashed_post = False
    
    for index, group in enumerate(EXPERIMENT_GROUPS):
        base_dir = group['base_dir']
        
        if not os.path.exists(base_dir):
            logging.error(f"Directory missing: '{base_dir}' does not exist. Skipping group {index}.")
            continue
            
        csv_path = os.path.join(base_dir, 'kinematics_data.csv')
        
        if not os.path.exists(csv_path):
            logging.warning(f"Kinematics data not found in {base_dir}. Skipping...")
            continue
            
        try:
            df_sampled = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Error reading CSV in {base_dir}: {e}")
            continue
            
        if df_sampled.empty:
            logging.warning(f"Kinematics data is empty in {base_dir}. Skipping...")
            continue
            
        logging.info(f"Successfully loaded data for group {index}: {group['label']}")
            
        color = group['color']
        marker = group['marker']
        label = group['label']
        
        breakthrough_df = df_sampled[df_sampled['Extruded_Area'] > 0]
        
        if not breakthrough_df.empty:
            cutoff_step = np.asarray(breakthrough_df['Step'])[0]
            # 共用关键节点防止多段线断裂
            df_pre = df_sampled[df_sampled['Step'] <= cutoff_step]
            df_post = df_sampled[df_sampled['Step'] >= cutoff_step] 
        else:
            df_pre = df_sampled
            df_post = pd.DataFrame(columns=df_sampled.columns)
            
        if not df_pre.empty:
            _ = ax.plot(df_pre['Shortening_km'], df_pre['Aspect_Ratio_Smooth'], 
                        color=color, marker=marker, markersize=8, 
                        markerfacecolor=color, markeredgecolor=color,
                        linestyle='-', linewidth=2, label=label)
                    
        if not df_post.empty:
            has_dashed_post = True
            _ = ax.plot(df_post['Shortening_km'], df_post['Aspect_Ratio_Smooth'], 
                        color=color, marker='', 
                        linestyle='--', linewidth=2)

    handles, labels = ax.get_legend_handles_labels()
    
    if has_dashed_post:
        proxy_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Salt extrusion (dashed lines)')
        handles.append(proxy_line)
        labels.append(proxy_line.get_label())
    
    # 强制将图例锚点归位且拒绝边框
    _ = ax.legend(handles, labels, loc='upper left', frameon=False)
    _ = ax.margins(x=0.15)
    
    plt.tight_layout()
    
    plot_png = os.path.join(FINAL_OUTPUT_DIR, 'Multi_Kinematic_Evolution_Plot.png')
    plot_pdf = os.path.join(FINAL_OUTPUT_DIR, 'Multi_Kinematic_Evolution_Plot.pdf')
    
    plt.savefig(plot_png, dpi=600, bbox_inches='tight')
    plt.savefig(plot_pdf, dpi=600, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Multi-comparison plot generated successfully in {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
