"""
ZDEM Salt Kinematics 交互式质控与边界修正系统

职责: 提供专家级的人机交互界面，用于手动校准自动算法可能误判的盐底辟边界基点。
工程化改进: 接入统一日志、术语规范化、优化 UI 交互逻辑与平滑算法、适配最新的视觉风格。
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any

from config import *
from utils import (
    apply_savgol_filter, 
    GroupDataManager, 
    ProfileData, 
    setup_project_logging
)

# 统一初始化日志
setup_project_logging()

class ManualCorrectorApp:
    """
    交互式修正应用程序类。
    """
    def __init__(self, mgr: GroupDataManager):
        self.mgr = mgr
        self.label = mgr.label
        
        # 数据加载与完整性校验
        try:
            self.df = pd.read_csv(mgr.csv_path)
            with open(mgr.pkl_path, 'rb') as f:
                self.profiles: Dict[int, ProfileData] = pickle.load(f)
        except Exception as e:
            logging.error(f"加载组别 [{self.label}] 缓存数据失败: {e}")
            raise
        
        self.steps = sorted(self.profiles.keys())
        self.current_idx = 0
        self._setup_ui()
        self._bind_events()
        self.update_plot()

    def _setup_ui(self):
        """初始化学术级极简 UI 布局。"""
        self.fig = plt.figure(figsize=(12, 7))
        # 布局分配：左侧主图，右侧信息看板，底部滑块
        gs = GridSpec(2, 2, height_ratios=[1, 0.15], width_ratios=[1, 0.22], hspace=0.3, wspace=0.1)
        
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.axis('off')
        
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        self.ax_slider.set_facecolor('#F8F9FA')
        
        # 标题与操作指南
        self.fig.text(0.1, 0.95, f"专家级边界质控协议 | 实验组: {self.label}", 
                     fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_main'])
        self.fig.text(0.1, 0.91, "快捷键: [←/→] 切换时间步 | [S/Enter] 保存并同步曲线 | [点击图中颗粒表面] 修正基点位置", 
                     fontsize=9, color=COLOR_PALETTE['text_sub'])

        # 交互滑块
        self.slider = Slider(
            ax=self.ax_slider,
            label='时间演化步 (Step) ',
            valmin=0,
            valmax=len(self.steps) - 1,
            valinit=0,
            valstep=1,
            color=COLOR_PALETTE['primary']
        )
        self.slider.label.set_size(10)
        
        # 信息看板文字初始化
        self.info_text = self.ax_info.text(0, 0.85, "", transform=self.ax_info.transAxes, 
                                          verticalalignment='top', fontsize=10, linespacing=1.8,
                                          fontfamily='monospace')

    def _bind_events(self):
        """绑定键盘与鼠标交互事件。"""
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
        """处理鼠标点击：实现盐体边缘基点的精确重定位。"""
        if event.inaxes == self.ax and event.xdata is not None:
            step = self.steps[self.current_idx]
            prof = self.profiles[step]
            
            # 计算点击位置与颗粒表面的欧氏距离，锁定最近采样点
            click_x = float(event.xdata)
            x_surf = np.asarray(prof['x'])
            y_smooth = apply_savgol_filter(np.asarray(prof['y']), EXTRACT_SMOOTH_WINDOW)
            
            idx = np.argmin(np.abs(x_surf - click_x))
            new_base_x = float(x_surf[idx])
            new_base_y = float(y_smooth[idx])
            
            # 更新缓存中的基点坐标
            self.profiles[step]['base_x'] = new_base_x
            self.profiles[step]['base_y'] = new_base_y
            
            # 同步更新指标 DataFrame，重新计算几何运动学参数
            top_x, top_y = float(prof.get('top_x', np.nan)), float(prof.get('top_y', np.nan))
            
            if not np.isnan(top_x):
                new_width = abs(top_x - new_base_x)
                new_relief = top_y - new_base_y
                new_ar = new_relief / new_width if new_width > 0 else np.nan
                
                mask = self.df['Step'] == step
                self.df.loc[mask, 'Width'] = new_width
                self.df.loc[mask, 'Relief'] = new_relief
                self.df.loc[mask, 'Aspect_Ratio'] = new_ar
                
                logging.info(f"人工干预 [Step {step}] -> 重定位基点: ({new_base_x:.1f}, {new_base_y:.1f}) | 新半宽: {new_width:.1f}m")
            
            self.update_plot()

    def update_plot(self):
        """实时刷新绘图区域与看板数据。"""
        step = self.steps[self.current_idx]
        prof = self.profiles[step]
        
        x, y = np.asarray(prof['x']), np.asarray(prof['y'])
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.4, color=COLOR_PALETTE['grid'], zorder=0)

        if len(x) == 0:
            self.ax.text(0.5, 0.5, f"Step {step}: 颗粒数据缺失", transform=self.ax.transAxes, ha='center')
        else:
            y_smooth = apply_savgol_filter(y, EXTRACT_SMOOTH_WINDOW)
            
            # 1. 盐体包络线渲染
            self.ax.plot(x, y_smooth, color=COLOR_PALETTE['primary'], lw=2.0, zorder=3, label='盐体颗粒表面')
            self.ax.fill_between(x, MANUAL_PLOT_Y_MIN, y_smooth, color=COLOR_PALETTE['primary'], alpha=0.12, zorder=2)
            
            # 2. 特征特征点标注
            if not np.isnan(prof['top_x']):
                self.ax.scatter(prof['top_x'], prof['top_y'], marker='*', s=220, color='#FFD700', 
                               edgecolors=COLOR_PALETTE['text_main'], zorder=6, label='盐丘主峰')
            if not np.isnan(prof['base_x']):
                self.ax.scatter(prof['base_x'], prof['base_y'], marker='o', s=120, color=COLOR_PALETTE['secondary'], 
                               edgecolors='white', linewidth=1.5, zorder=6, label='人工修正基点')

        # 3. 坐标轴与样式配置
        self.ax.set_xlim(0, MODEL_WIDTH)
        self.ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
        self.ax.set_xlabel("水平向距离 (m)", color=COLOR_PALETTE['text_sub'])
        self.ax.set_ylabel("高程 (m)", color=COLOR_PALETTE['text_sub'])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(loc='upper right', frameon=False, fontsize=9)
        
        # 4. 看板指标同步显示
        df_row = self.df[self.df['Step'] == step]
        shortening = df_row['Shortening_km'].values[0] if not df_row.empty else 0
        width = abs(prof['top_x'] - prof['base_x']) if not np.isnan(prof['base_x']) else 0
        relief = prof['top_y'] - prof['base_y'] if not np.isnan(prof['base_x']) else 0
        ar = relief / width if width > 0 else 0
        
        info_str = (
            f"  [物理演化看板]\n"
            f"  ━━━━━━━━━━━━━\n"
            f"  时间步: {step}\n"
            f"  缩短量: {shortening:.4f} km\n\n"
            f"  核心指标 (M):\n"
            f"  颗粒半宽: {width:>6.1f}\n"
            f"  地形起伏: {relief:>6.1f}\n"
            f"  高宽比  : {ar:>8.4f}"
        )
        self.info_text.set_text(info_str)
        self.fig.canvas.draw_idle()

    def run(self):
        """启动 GUI 循环。"""
        plt.show()
        self._finalize()

    def _finalize(self):
        """退出前执行二次平滑并持久化修正结果。"""
        # 重新应用平滑滤波，确保修正后的突变点得到平抑
        for col in ['Aspect_Ratio', 'Width', 'Relief']:
            if col in self.df.columns:
                self.df[f'{col}_Smooth'] = self.df[col].rolling(
                    window=int(SMOOTHING_WINDOW), min_periods=1, center=True
                ).mean()
        
        self.df.to_csv(self.mgr.csv_path, index=False)
        with open(self.mgr.pkl_path, 'wb') as f:
            pickle.dump(self.profiles, f)
        logging.info(f"实验组 [{self.label}] 的人工修正数据已同步，并完成二次曲线平滑。")

def main():
    """程序入口。"""
    logging.info("========== 启动 ZDEM 盐底辟运动学 QA/QC 交互系统 ==========")
    for group in EXPERIMENT_GROUPS:
        mgr = GroupDataManager(group)
        if not os.path.exists(mgr.csv_path) or not os.path.exists(mgr.pkl_path):
            logging.warning(f"跳过组别 [{mgr.label}]: 找不到提取的缓存文件。")
            continue

        logging.info(f"正在加载实验组 [{mgr.label}] 数据...")
        app = ManualCorrectorApp(mgr)
        app.run()

if __name__ == '__main__':
    main()
