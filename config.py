import os
import numpy as np

# ==========================================
# 1. 核心目录与实验组配置
# ==========================================
FINAL_OUTPUT_DIR = r"E:\0.Information\4.Temp\StructLab\盐构造部分\实验\68"

EXPERIMENT_GROUPS = [
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '150'), 'label': r'$v_e = 150 \ m \cdot s^{-1}$', 'color': 'b', 'marker': 'o'},
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '300'), 'label': r'$v_e = 300 \ m \cdot s^{-1}$', 'color': 'r', 'marker': 's'},
    {'base_dir': os.path.join(FINAL_OUTPUT_DIR, '600'), 'label': r'$v_e = 600 \ m \cdot s^{-1}$', 'color': 'g', 'marker': '^'}
]

CSV_FILENAME = 'kinematics_data.csv'
PKL_FILENAME = 'profiles_cache.pkl'

# ==========================================
# 2. 物理模型尺寸与绘图视口配置 (Viewport)
# ==========================================
MODEL_WIDTH = 80000.0         # 模型总水平宽度 (m)
MODEL_HEIGHT = 30000.0        # 模型总绝对高度 (m)

MANUAL_PLOT_Y_MIN = 5000.0    # 01b 交互修正时的 Y 轴下限 (m)
MANUAL_PLOT_Y_MAX = 25000.0   # 01b 交互修正时的 Y 轴上限 (m)

MAX_SHORTENING_KM = 24.0      # 最大演化挤压量 (km) (图表的 X 轴上限)
MAX_ASPECT_RATIO = 0.30       # 演化曲线高宽比 Y 轴上限

# ==========================================
# 3. 离散元物理参数与算法阈值
# ==========================================
SALT_GROUP_NAME = 'salt'
PARTICLE_RADIUS = 70.0        # 颗粒半径
PARTICLE_AREA = np.pi * (PARTICLE_RADIUS ** 2)
NUM_BINS = 200                # 形态提取网格数
SMOOTHING_WINDOW = 5          # 最终曲线平滑窗口 (用于 01, 01b)
EXTRACT_SMOOTH_WINDOW = 51    # 形态初始提取平滑窗口 (用于 01)
NUM_KEY_STAGES = 10           # 强制采样关键帧数量
