# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# Matplotlib 中文渲染配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 全局配置与工作环境初始化
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from config import *

def main():
    # 遍历实验组
    for group in EXPERIMENT_GROUPS:
        target_base_dir = group['base_dir']
        group_label = os.path.basename(target_base_dir)
        
        logging.info(f"\n{'='*50}\n========== 开始人工核对组别: {target_base_dir} ==========\n{'='*50}")
        
        csv_path = os.path.join(target_base_dir, CSV_FILENAME)
        pkl_path = os.path.join(target_base_dir, PKL_FILENAME)

        if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
            logging.warning(f"文件缺失！组别 {group_label} 下不存在 {CSV_FILENAME} 或 {PKL_FILENAME}，跳过该组。")
            continue

        # 加载序列化数据
        logging.info(f"正在读取组别 [{group_label}] 的数据...")
        df = pd.read_csv(csv_path)
        with open(pkl_path, 'rb') as f:
            profiles_cache = pickle.load(f)

        # 按时间步升序排列
        sorted_steps = sorted(profiles_cache.keys())
        
        # 启用交互模式
        plt.ion()

        # ==========================================
        # 2. 交互式可视化与数据遍历拾取 (内层循环)
        # ==========================================
        for step in sorted_steps:
            prof_data = profiles_cache[step]
            
            # 提取剖面坐标并强制类型转换
            x_surf = np.asarray(prof_data['x'], dtype=float)
            raw_y = np.asarray(prof_data['y'], dtype=float)
            
            # Savitzky-Golay 平滑盐层顶面剖面
            window_len = min(EXTRACT_SMOOTH_WINDOW, len(raw_y))
            if window_len % 2 == 0: window_len -= 1
            y_smooth = np.asarray(savgol_filter(raw_y, window_length=window_len, polyorder=3), dtype=float)
            
            top_x = prof_data['top_x']
            top_y = prof_data['top_y']
            base_x = prof_data['base_x']
            base_y = prof_data['base_y']

            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制盐层顶面剖面
            ax.plot(x_surf, y_smooth, 'r-', linewidth=2, label='盐丘上表面 (Salt Profile)')
            ax.fill_between(x_surf, np.min(y_smooth), y_smooth, color='pink', alpha=0.4)
            
            # 标注自动识别的特征点
            ax.plot(top_x, top_y, 'r*', markersize=18, label='主峰点 (Central Peak)')
            ax.plot(base_x, base_y, 'bv', markersize=12, label='算法预估非挤压端基点 (Auto Base)')

            ax.set_title(f"【{group_label}】 Step {step} — 交互式基点修正\n"
                         f"接受: 关闭窗口或按 Enter 跳过 | "
                         f"修正: 点击非挤压端基点后按 Enter 确认 (取最后一次点击)", fontsize=13)
            ax.set_xlabel('水平距离 (m)')
            ax.set_ylabel('高程 (m)')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 固定视口范围
            ax.set_xlim(0, MODEL_WIDTH)
            ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
            
            plt.tight_layout()
            plt.show(block=False)

            # ==========================================
            # 3. 阻塞式监听与用户意图判定
            # ==========================================
            try:
                # 阻塞式无限次点击监听，回车键终止
                clicks = plt.ginput(n=-1, timeout=0, show_clicks=True)
            except Exception as e:
                # 窗口关闭等中断异常处理
                logging.debug(f"窗口被强制关闭或其他中断: {e}")
                clicks = []
                
            plt.close(fig)

            # 存在有效点击 → 提取最后一次点击坐标作为修正基点
            if len(clicks) >= 1:
                click_x, click_y = clicks[-1]
                
                # 坐标吸附：锁定最近离散采样点
                new_idx = int(np.argmin(np.abs(x_surf - click_x)))

                # 基于平滑曲线获取吸附坐标
                new_base_x, new_base_y = x_surf[new_idx], y_smooth[new_idx]

                # 基于修正基点重算形态参数
                new_relief = top_y - new_base_y
                new_width = abs(top_x - new_base_x)
                
                if new_width > 0:
                    new_aspect_ratio = new_relief / new_width
                    
                    # 获取修正前原值用于日志对比
                    mask = df['Step'] == step
                    old_ar = df.loc[mask, 'Aspect_Ratio'].values[0] if mask.any() else np.nan
                    
                    logging.info(f"==> 组别 {group_label} | Step {step} 修正成功 | 高宽比变更: {old_ar:.4f} -> {new_aspect_ratio:.4f} | 新宽度: {new_width:.1f}m")

                    # 同步更新缓存与 DataFrame
                    profiles_cache[step]['y'] = y_smooth
                    profiles_cache[step]['base_x'] = new_base_x
                    profiles_cache[step]['base_y'] = new_base_y
                    
                    if mask.any():
                        df.loc[mask, 'Aspect_Ratio'] = new_aspect_ratio
                        if 'Width' in df.columns:
                            df.loc[mask, 'Width'] = new_width
                        if 'Relief' in df.columns:
                            df.loc[mask, 'Relief'] = new_relief
            else:
                logging.info(f"组别 {group_label} | Step {step} 用户选择跳过。保留原自动化算法特征。")

        # ==========================================
        # 4. 后处理平滑与落盘保存 (组循环内部，每组处理完即时落盘)
        # ==========================================
        plt.ioff()
        logging.info(f"组别 [{group_label}] 所有 Step 迭代完毕，开始进行最后的数据清洗和存盘...")

        # 滚动均值平滑（保留 NaN 不扩散）
        if 'Aspect_Ratio' in df.columns:
            df['Aspect_Ratio_Smooth'] = df['Aspect_Ratio'].rolling(
                window=int(SMOOTHING_WINDOW), 
                min_periods=1, 
                center=True
            ).mean()
            
        if 'Width' in df.columns and 'Relief' in df.columns:
            df['Width_Smooth'] = df['Width'].rolling(
                window=int(SMOOTHING_WINDOW), 
                min_periods=1, 
                center=True
            ).mean()
            df['Relief_Smooth'] = df['Relief'].rolling(
                window=int(SMOOTHING_WINDOW), 
                min_periods=1, 
                center=True
            ).mean()

        # 覆写序列化文件
        df.to_csv(csv_path, index=False)
        with open(pkl_path, 'wb') as f:
            pickle.dump(profiles_cache, f)
            
        logging.info(f"组别 [{group_label}] 人工修正与平滑操作完成！数据已安全写入 {target_base_dir}\n")

    logging.info("所有实验组的数据人工核对任务全部结束！")

if __name__ == '__main__':
    main()