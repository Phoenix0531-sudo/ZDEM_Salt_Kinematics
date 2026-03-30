# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedExpression=false
import os
import pickle
import logging
import numpy as np
import pandas as pd
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

from config import *

# ==========================================
# 2. 诊断图表渲染核心
# ==========================================
def main():
    # 遍历所有配置的实验组
    for index, group in enumerate(EXPERIMENT_GROUPS):
        base_dir = group['base_dir']
        label = group['label']
        
        if not os.path.exists(base_dir):
            logging.error(f"Directory missing: '{base_dir}' does not exist. Skipping group {index}.")
            continue
            
        logging.info(f"Processing group index: {index}. Checking data directory: {base_dir}")
        
        csv_path = os.path.join(base_dir, 'kinematics_data.csv')
        pkl_path = os.path.join(base_dir, 'profiles_cache.pkl')
        
        if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
            missing_path = csv_path if not os.path.exists(csv_path) else pkl_path
            logging.error(f"Required data files not found. Please execute 01_data_extractor.py first. Missing path: {missing_path}")
            continue
            
        try:
            df_sampled = pd.read_csv(csv_path)
            with open(pkl_path, 'rb') as f:
                profiles_data_store = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading data for {base_dir}: {e}")
            continue

        logging.info("Data loaded successfully. Rendering diagnostic plots...")

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 1.5 
        
        # -----------------------------------------------------
        # 图 1：极简主折线图
        # -----------------------------------------------------
        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('white')  
        ax.grid(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='x', direction='in', top=False, bottom=True, length=6, width=1.5)
        ax.tick_params(axis='y', direction='in', left=False, right=True, length=6, width=1.5)
        
        _ = ax.set_xlim(0, MAX_SHORTENING_KM)
        _ = ax.set_ylim(0, MAX_ASPECT_RATIO)
        _ = ax.set_xlabel('Shortening (km)', weight='bold') 
        _ = ax.set_ylabel('Aspect ratio', weight='bold') 
        
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
            _ = ax.plot(df_pre['Shortening_km'], df_pre['Aspect_Ratio_Smooth'], 
                        color=academic_blue, marker='o', markersize=9, 
                        markerfacecolor=academic_blue, markeredgecolor=academic_blue,
                        linestyle='-', linewidth=2, label=label)
                    
        if not df_post.empty and not df_pre.empty:
            _ = ax.plot(df_post['Shortening_km'], df_post['Aspect_Ratio_Smooth'], color=academic_blue, linestyle='--', linewidth=2)
            
        handles, labels_list = ax.get_legend_handles_labels()
        if not df_post.empty:
            proxy_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Salt extrusion (dashed line)')
            handles.append(proxy_line)
            labels_list.append(proxy_line.get_label())
        
        if handles:
            _ = ax.legend(handles, labels_list, loc='upper left', frameon=False)
        
        _ = ax.margins(x=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'Kinematic_Evolution_Diagnostic.png'), dpi=600, bbox_inches='tight')
        plt.close() 

        # -----------------------------------------------------
        # 图 2：盐体剖面形态演化多宫格诊断图
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
                
                _ = ax_p.fill_between(p_data['x'], 0, p_data['y'], color='lightpink', alpha=0.8, label='Salt Body')
                _ = ax_p.plot(p_data['x'], p_data['y'], color='crimson', linewidth=2)
                _ = ax_p.scatter([p_data['top_x']], [p_data['top_y']], color='red', marker='*', s=200, zorder=5)
                _ = ax_p.scatter([p_data['base_x']], [p_data['base_y']], color='blue', marker='v', s=100, zorder=5)
                
                shortening_arr = np.asarray(df_sampled[df_sampled['Step'] == step]['Shortening_km'])
                shortening = shortening_arr[0] if len(shortening_arr) > 0 else 0.0
                _ = ax_p.set_title(f'Step: {step} | Shortening: {shortening:.1f} km', fontweight='bold', fontsize=12)
                
                _ = ax_p.set_xlim(0, MODEL_WIDTH)
                _ = ax_p.set_ylim(0, MODEL_HEIGHT)
                _ = ax_p.tick_params(labelsize=10)
                
            for i in range(num_plots, len(axes_prof)):
                axes_prof[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, 'Salt_Profiles_Diagnostic_Grid.png'), dpi=300, bbox_inches='tight')
            plt.close()

        logging.info(f"Diagnostic plots generated and saved to: {base_dir}")

    logging.info("All groups processed successfully.")

if __name__ == '__main__':
    main()