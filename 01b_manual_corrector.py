# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 【新增】引入平滑滤波库

# ==========================================
# 解决 Matplotlib 中文显示乱码问题 【新增】
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 优先使用系统黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 确保负号正常显示

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
    # 外层循环包裹：遍历所有的目标实验组
    for group in EXPERIMENT_GROUPS:
        target_base_dir = group['base_dir']
        group_label = os.path.basename(target_base_dir)
        
        logging.info(f"\n{'='*50}\n========== 开始人工核对组别: {target_base_dir} ==========\n{'='*50}")
        
        csv_path = os.path.join(target_base_dir, CSV_FILENAME)
        pkl_path = os.path.join(target_base_dir, PKL_FILENAME)

        if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
            logging.warning(f"文件缺失！组别 {group_label} 下不存在 {CSV_FILENAME} 或 {PKL_FILENAME}，跳过该组。")
            continue

        # 加载持久化数据
        logging.info(f"正在读取组别 [{group_label}] 的数据...")
        df = pd.read_csv(csv_path)
        with open(pkl_path, 'rb') as f:
            profiles_cache = pickle.load(f)

        # 按照 Step 从小到大排序，确保时间序列的连贯性
        sorted_steps = sorted(profiles_cache.keys())
        
        # 开启 Matplotlib 交互模式
        plt.ion()

        # ==========================================
        # 2. 交互式可视化与数据遍历拾取 (内层循环)
        # ==========================================
        for step in sorted_steps:
            prof_data = profiles_cache[step]
            
            # 【核心修改区：加入物理熨斗平滑】
            # 解析解构缓存中的坐标信息，并强制转化为安全的 float numpy 数组以兼容检查
            x_surf = np.asarray(prof_data['x'], dtype=float)
            raw_y = np.asarray(prof_data['y'], dtype=float)
            
            # 动态增强平滑：采用全局统一的强力平滑窗口，彻底抹掉离散元毛刺
            window_len = min(EXTRACT_SMOOTH_WINDOW, len(raw_y))
            if window_len % 2 == 0: window_len -= 1
            y_smooth = np.asarray(savgol_filter(raw_y, window_length=window_len, polyorder=3), dtype=float)
            
            top_x = prof_data['top_x']
            top_y = prof_data['top_y']
            base_x = prof_data['base_x']
            base_y = prof_data['base_y']

            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制盐底的几何轮廓
            ax.plot(x_surf, y_smooth, 'r-', linewidth=2, label='盐丘上表面 (Salt Profile)')
            ax.fill_between(x_surf, np.min(y_smooth), y_smooth, color='pink', alpha=0.4)
            
            # 标绘算法原先预估的几何特征点
            ax.plot(top_x, top_y, 'r*', markersize=18, label='主峰点 (Central Peak)')
            ax.plot(base_x, base_y, 'bv', markersize=12, label='算法预估非挤压端基点 (Auto Base)')

            ax.set_title(f"【组别: {group_label}】 Step {step} - 人工修正干预系统 (QA/QC)\n"
                         f"【满意】直接关闭窗口或按回车跳过。\n"
                         f"【修正】请在真实的『非挤压端基底』点击一个基点，然后按【回车键 (Enter)】确认。\n"
                         f"如果点错了，可以继续点击覆盖，程序只取最后一次点击。", fontsize=13)
            ax.set_xlabel('水平距离 (m)')
            ax.set_ylabel('高程 (m)')
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 绝对固定的坐标轴视口 (防自适应抖动)
            ax.set_xlim(0, MODEL_WIDTH)
            ax.set_ylim(MANUAL_PLOT_Y_MIN, MANUAL_PLOT_Y_MAX)
            
            plt.tight_layout()
            plt.show(block=False)

            # ==========================================
            # 3. 阻塞式监听与用户意图判定
            # ==========================================
            try:
                # 挂起程序，允许无限次点击 (n=-1)，直到用户按下回车键，防误触
                clicks = plt.ginput(n=-1, timeout=0, show_clicks=True)
            except Exception as e:
                # 用户强行关闭图形窗口等中断异常
                logging.debug(f"窗口被强制关闭或其他中断: {e}")
                clicks = []
                
            plt.close(fig)  # 清理当前图形，释放内存

            # 如果点击数 >= 1，说明用户想修正，提取最后一次点击作为基底边界点
            if len(clicks) >= 1:
                click_x, click_y = clicks[-1]
                
                # 坐标吸附 (Snapping) - 寻找距离用户点击光标最接近的真实数组点位
                # 强制转换为 int 类型以避免 Linter 关于 intp 类型不能做索引的错误预警
                new_idx = int(np.argmin(np.abs(x_surf - click_x)))

                # 注意：这里使用的是平滑后的 y_smooth，让你的点击吸附在平滑曲线上
                new_base_x, new_base_y = x_surf[new_idx], y_smooth[new_idx]

                # 特征值重算机制 (Asymmetric Single-Anchor)
                new_relief = top_y - new_base_y
                new_width = abs(top_x - new_base_x)
                
                if new_width > 0:
                    new_aspect_ratio = new_relief / new_width
                    
                    # 记录修正前的原始数据用于对比显示
                    mask = df['Step'] == step
                    old_ar = df.loc[mask, 'Aspect_Ratio'].values[0] if mask.any() else np.nan
                    
                    logging.info(f"==> 组别 {group_label} | Step {step} 修正成功 | 高宽比变更: {old_ar:.4f} -> {new_aspect_ratio:.4f} | 新宽度: {new_width:.1f}m")

                    # 更新持久化缓存字典和 DataFrame 内存
                    # 【核心修复】同步更新 y 坐标数据，确保 02 脚本绘图时的底图与点击时的底图完全一致
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

        # 单侧或全局高宽比平滑：采用 Rolling Mean (中心对称窗口)
        # 此处必须严格保证 Aspect_Ratio 列中原本可能存在的 NaN 不会被无限扩大，使用 min_periods=1
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

        # 覆写保存回原文件，确保下一步多组绘图脚本可以调用最新的结果
        df.to_csv(csv_path, index=False)
        with open(pkl_path, 'wb') as f:
            pickle.dump(profiles_cache, f)
            
        logging.info(f"组别 [{group_label}] 人工修正与平滑操作完成！数据已安全写入 {target_base_dir}\n")

    logging.info("所有实验组的数据人工核对任务全部结束！")

if __name__ == '__main__':
    main()