"""
ZDEM Salt Kinematics 交互式质控与边界修正系统 (V2.0)

职责: 提供专家级的人机交互界面，用于手动校准自动算法可能误判的盐丘边界基点。
特色: 学术级极简视觉风格、类封装架构、键盘快捷键支持、实时数据看板。
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from typing import TypedDict, List, Dict, Optional, Any

from config import *
from utils import apply_savgol_filter

# ==========================================
# 1. 类型定义与全局样式
# ==========================================
class ProfileData(TypedDict):
    step: int
    x: np.ndarray
    y: np.ndarray
    top_x: float
    top_y: float
    base_x: float
    base_y: float

# 学术级配色方案 (Deep & Minimal)
COLOR_PALETTE = {
    'primary': '#B2182B',    # 盐丘主色 (深红)
    'secondary': '#4C72B0',  # 基点主色 (靛蓝)
    'background': '#FFFFFF', # 背景
    'grid': '#E0E0E0',      # 网格
    'text_main': '#212121',  # 主文字
    'text_sub': '#757575',   # 辅助文字
    'accent': '#FF8F00'      # 强调色
}

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'Arial', 'sans-serif'],
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'font.size': 10,
    'figure.facecolor': COLOR_PALETTE['background']
})

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class ManualCorrectorApp:
    def __init__(self, group_config: Dict[str, Any], csv_path: str, pkl_path: str):
        self.group = group_config
        self.csv_path = csv_path
        self.pkl_path = pkl_path
        self.label = os.path.basename(group_config['base_dir'])
        
        # 数据初始化
        self.df = pd.read_csv(csv_path)
        with open(pkl_path, 'rb') as f:
            self.profiles: Dict[int, ProfileData] = pickle.load(f)
        
        self.steps = sorted(self.profiles.keys())
        self.current_idx = 0
        self.history: List[Dict[str, Any]] = [] # 用于撤销 (可选)

        # 构建 UI
        self._setup_ui()
        self._bind_events()
        self.update_plot()

    def _setup_ui(self):
        """初始化学术级布局。"""
        self.fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(2, 2, height_ratios=[1, 0.15], width_ratios=[1, 0.2], hspace=0.3, wspace=0.1)
        
        # 主绘图区
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1]) # 侧边信息栏
        self.ax_info.axis('off')
        
        # 底部控件区
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        self.ax_slider.set_facecolor('#F5F5F5')
        
        # 顶部标题与说明
        self.fig.text(0.1, 0.95, f"专家质控协议 | 实验组: {self.label}", fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_main'])
        self.instruction_text = self.fig.text(0.1, 0.91, "快捷键: [←/→] 切换帧 | [S/Enter] 保存并关闭 | [点击图内] 重定位基点", 
                                             fontsize=9, color=COLOR_PALETTE['text_sub'])

        # 滑块
        self.slider = Slider(
            ax=self.ax_slider,
            label='时间序列 ',
            valmin=0,
            valmax=len(self.steps) - 1,
            valinit=0,
            valstep=1,
            color=COLOR_PALETTE['primary'],
            initcolor='none'
        )
        self.slider.label.set_size(10)
        
        # 侧边看板文字初始化
        self.info_text = self.ax_info.text(0, 0.8, "", transform=self.ax_info.transAxes, verticalalignment='top', fontsize=10, linespacing=1.8)

    def _bind_events(self):
        """绑定交互事件。"""
        self.slider.on_changed(self._on_slider_change)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_slider_change(self, val):
        self.current_idx = int(val)
        self.update_plot()

    def _on_key(self, event):
        if event.key == 'right':
            next_idx = min(self.current_idx + 1, len(self.steps) - 1)
            self.slider.set_val(next_idx)
        elif event.key == 'left':
            prev_idx = max(self.current_idx - 1, 0)
            self.slider.set_val(prev_idx)
        elif event.key in ['enter', 's']:
            plt.close(self.fig)

    def _on_click(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            step = self.steps[self.current_idx]
            prof = self.profiles[step]
            
            # 计算最近的点
            click_x = float(event.xdata)
            x_surf = np.asarray(prof['x'])
            y_smooth = apply_savgol_filter(np.asarray(prof['y']), EXTRACT_SMOOTH_WINDOW)
            
            idx = np.argmin(np.abs(x_surf - click_x))
            new_base_x = float(x_surf[idx])
            new_base_y = float(y_smooth[idx])
            
            # 更新 Profile 数据
            self.profiles[step]['base_x'] = new_base_x
            self.profiles[step]['base_y'] = new_base_y
            
            # 同步更新 DataFrame
            top_x = float(prof.get('top_x', np.nan))
            top_y = float(prof.get('top_y', np.nan))
            
            if not np.isnan(top_x):
                new_width = abs(top_x - new_base_x)
                new_relief = top_y - new_base_y
                new_aspect_ratio = new_relief / new_width if new_width > 0 else np.nan
                
                mask = self.df['Step'] == step
                self.df.loc[mask, 'Width'] = new_width
                self.df.loc[mask, 'Relief'] = new_relief
                self.df.loc[mask, 'Aspect_Ratio'] = new_aspect_ratio
                
                logging.info(f"Step {step} 已手动修正: Width={new_width:.1f}m, AR={new_aspect_ratio:.4f}")
            
            self.update_plot()

    def update_plot(self):
        """刷新画面与看板。"""
        step = self.steps[self.current_idx]
        prof = self.profiles[step]
        
        # 数据准备
        x = np.asarray(prof['x'])
        y = np.asarray(prof['y'])
        y_smooth = apply_savgol_filter(y, EXTRACT_SMOOTH_WINDOW)
        
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.5, color=COLOR_PALETTE['grid'], zorder=0)
        
        # 1. 绘制主体
        self.ax.plot(x, y_smooth, color=COLOR_PALETTE['primary'], lw=1.8, zorder=3, label='盐丘表面')
        self.ax.fill_between(x, np.min(y_smooth), y_smooth, color=COLOR_PALETTE['primary'], alpha=0.1, zorder=2)
        
        # 2. 绘制特征点
        if not np.isnan(prof['top_x']):
            self.ax.scatter(prof['top_x'], prof['top_y'], marker='*', s=180, color=COLOR_PALETTE['primary'], 
                           edgecolors='white', zorder=5, label='主峰锚点')
        if not np.isnan(prof['base_x']):
            self.ax.scatter(prof['base_x'], prof['base_y'], marker='o', s=100, color=COLOR_PALETTE['secondary'], 
                           edgecolors='white', zorder=5, label='修正基点')

        # 3. 轴样式
        self.ax.set_xlim(0, MODEL_WIDTH)
        self.ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
        self.ax.set_xlabel("水平向距离 (m)", color=COLOR_PALETTE['text_sub'])
        self.ax.set_ylabel("高程 (m)", color=COLOR_PALETTE['text_sub'])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(loc='upper right', frameon=False, fontsize=9)
        
        # 4. 看板更新
        df_row = self.df[self.df['Step'] == step]
        shortening = df_row['Shortening_km'].values[0] if not df_row.empty else 0
        width = prof['top_x'] - prof['base_x'] if not np.isnan(prof['base_x']) else 0
        relief = prof['top_y'] - prof['base_y'] if not np.isnan(prof['base_x']) else 0
        ar = relief / abs(width) if width != 0 else 0
        
        info_str = (
            f"物理状态看板\n"
            f"━━━━━━━━━━━━\n"
            f"帧号: {step}\n"
            f"缩短量: {shortening:.3f} km\n\n"
            f"运动学参数:\n"
            f"宽度: {abs(width):.1f} m\n"
            f"起伏: {relief:.1f} m\n"
            f"高宽比: {ar:.4f}"
        )
        self.info_text.set_text(info_str)
        
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()
        self._finalize()

    def _finalize(self):
        """保存数据并执行平滑后处理。"""
        logging.info(f"正在同步 {self.label} 修正结果并执行平滑...")
        
        # 后处理平滑逻辑 (保持 NaN 结构)
        for col in ['Aspect_Ratio', 'Width', 'Relief']:
            if col in self.df.columns:
                self.df[f'{col}_Smooth'] = self.df[col].rolling(
                    window=int(SMOOTHING_WINDOW), min_periods=1, center=True
                ).mean()
        
        self.df.to_csv(self.csv_path, index=False)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.profiles, f)
        logging.info("落盘完成。")

def main():
    for group in EXPERIMENT_GROUPS:
        target_base_dir = group['base_dir']
        csv_path = os.path.join(target_base_dir, CSV_FILENAME)
        pkl_path = os.path.join(target_base_dir, PKL_FILENAME)

        if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
            logging.warning(f"跳过组 {target_base_dir}: 文件缺失")
            continue

        app = ManualCorrectorApp(group, csv_path, pkl_path)
        app.run()

if __name__ == '__main__':
    main()
