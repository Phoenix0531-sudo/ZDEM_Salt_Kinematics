# ZDEM 离散元盐构造数值模拟：运动学提取与可视化管线

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![ZDEM](https://img.shields.io/badge/ZDEM-Simulation-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **摘要 (Abstract)**: 本体系提供了一套针对离散元 (ZDEM) 盐流变模拟的高吞吐量分析管线，旨在将海量原始空间物理帧数据以零损耗模式提取，并持久化为运动学几何特征曲线。该系统采用前端图形渲染与后台数据运算完全解耦的架构，并在管线中段集成嵌入了支持连续反馈的 "专家在环" 人机交互式 (QA/QC) 边界修正组件，最终实现高重现度的学术级出图与多组运动学指标对比。

## 1. 概述
本项目提供了一套专为 ZDEM (Z-Discrete Element Method) 盐岩构造模拟结果定制的 Python 数据分析与可视化管线。其核心科学目标在于，通过对模型演化过程中的离散颗粒几何空间信息提取与时间序列追踪，实现从庞大的原始数值解文件（`.dat`）向高维运动学指标演化曲线（如高宽比 Aspect Ratio、挤压量 Shortening 等）的高效降维转化。

最新版本大幅度重构了代码架构逻辑，将所有全局常量（软硬件参数）剥离至独立的集约配置文件 `config.py`，并创新性地加入了 `01b_manual_corrector.py` 专家干预辅助模块，以校正离散元计算法中颗粒游离所引发的非典型拓扑误判畸变。

## 2. 环境依赖 (Dependencies)
在部署此解析环境前，请确保 Python 解释器中已内置以下核心科学计算堆栈：
```bash
pip install numpy pandas scipy matplotlib tqdm
```

### 2.1 系统与硬件配置要求 (System Requirements)
ZDEM 的离散元计算帧数组通常占据巨量的存储带宽。鉴于本分析框架在初始提取阶应用了高并发多进程内存常驻调度机制，强烈推荐您的宿主节点达到或超过以下硬件配额：
- **操作系统**：推荐经内核优化的 Linux 发行版（如 Ubuntu 20.04 LTS 等）以获取最佳的系统级 IPC (进程间通信) 调度；同架构亦对 Windows 10/11 实现全核兼容。
- **处理器 (CPU)**：推荐采用 8 物理核心 16 逻辑线程及以上的架构（例如 AMD Ryzen 7 或 Intel Core i7 阵营）。执行池的工作单元上限将实质性决定由算法 `01` 发起的并行缩分效率。
- **内存堆栈 (RAM)**：为防范极端并发任务引发系统 OOM (Out Of Memory) 崩溃，本流程最低运行阈值需具备 **16GB** 物理内存（推荐标配 32GB 双通道），以支撑峰值时深度的特征数组常驻。

## 3. 核心文件与系统架构 (Architecture & Modules)

```mermaid
graph TD
    subgraph 数据降维与解析层 (Data Extraction Layer)
        A([原始 .dat 物理实验帧文件])
        B(01_data_extractor.py)
    end
    
    subgraph 专家在环与质控层 (Human-in-the-loop QA/QC Layer)
        C[(kinematics_data.csv <br> profiles_cache.pkl)]
        D(01b_manual_corrector.py)
    end
    
    subgraph 多维可视化渲染层 (Visualization Layer)
        E1(02_plot_single_diagnostic.py)
        E2(03_plot_multi_comparison.py)
        F1([单组诊断图谱输出])
        F2([终极跨组联合图谱输出])
    end

    A -->|高并发降维| B
    B -->|结构化张量持久化| C
    C -.->|QA/QC 强干预| D
    D -.->|人工基底截断与重平滑| C
    C -->|切片渲染引擎| E1
    C -->|联合出图引擎| E2
    E1 -->|多宫格与单组演化| F1
    E2 -->|灰阶代理学术化| F2

    classDef file fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef script fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    class A,F1,F2 file;
    class B,D,E1,E2 script;
    class C data;
```

该计算管线由独立的配置枢纽与四大协同运算内核组成：

### ⚙️ `config.py` (全局物理参数与配置中枢)
接管全计算链路的关键参数空间池化职责：
- **挂载与流转路径**：中心化封装 `FINAL_OUTPUT_DIR` 与目标参数系字典 `EXPERIMENT_GROUPS`。
- **空间域与图形成像边界**：锁定全局几何阈值 `MODEL_WIDTH`、`MODEL_HEIGHT`，提供严格锁死的视口防抖极值 `MANUAL_PLOT_Y_MIN/MAX`。
- **滤波解析算子**：固化系统全局窗长参数 (`SMOOTHING_WINDOW`)、统计栅格网格 (`NUM_BINS`)，以及核心交互几何约束的离散域基础元素半径 (`PARTICLE_RADIUS`)。

### 📌 `01_data_extractor.py` (后台并发数据提取器)
此脚本主掌底层核心多进程解析：
- **宏并发架构**：部署 `concurrent.futures.ProcessPoolExecutor` 服务，达成跨组寻址与组态内高密度多帧解耦化抽取流。
- **拓扑特征探测**：构建 `binned_statistic` 探针以逼近外边缘包络拟合；联立 Savitzky-Golay 低通数字滤波器及斜率约束搜寻器，精准确立特征几何位姿点（左/右基底锚位与顶部隆升主极值点）。
- **轻量化解耦与结构化持久存储**：完成降维后，将所有状态字典沉淀至一维 `kinematics_data.csv` 表和拓扑词典压缩包 `profiles_cache.pkl`，全程断开图形显卡与高密度渲染请求，实现极致速度解耦。

### 📌 `01b_manual_corrector.py` (交互式人工质控与修正系统)
连接自动化推演与精细学究干预的核心 QA/QC 桥梁：
- **人机协同**：循环挂载 Matplotlib 对冲界面，将控制权转接分析员，容许经由直观视图捕获系统误差。预先启用了高阶多项式滤波平滑 (e.g., Savitzky-Golay)，增强了人工识别视觉标定物的基线可靠度。
- **容错重采样与无感交互机制**：提供基于非拥塞的后入式截流功能（允许通过重复修正拾取光标置换早期失误点击），并严格验证回车事件的终态签封。遵循 “局部特征等势面弦截法则” 进行特征值收敛。
- **参数热更新与状态覆写**：针对修正点位的动力学宽幅与高宽标量进行即席更新，驱动移动窗口补偿以抑制时间序列振降，接着覆写源数据池 (`csv` / `pkl`)以完成自修正闭环。

### 📌 `02_plot_single_diagnostic.py` (单组剖面诊断图渲染器)
独立工作流形态演化审视：
- **演化切层检验**：遍历 `profiles_cache.pkl` 内核心帧关键节点，执行自动排列绘制并叠加生成含掩膜实体渲染的切片诊断阵列 `Salt_Profiles_Diagnostic_Grid.png`，完整标识动力学界限与核心顶点要素。
- **单系指纹图分析**：导出对应单独物理组别的宏周期演进表现图 `Kinematic_Evolution_Diagnostic.png`。

### 📌 `03_plot_multi_comparison.py` (多组联合演化对比展现)
出版级学术总视图的投射器：
- 基于字典配置自动检索全库各物理环境参数序列的可用性与拓扑特性，通过单一视图渲染合并所有群组动态趋势至 `Multi_Kinematic_Evolution_Plot.png` (并提供原生矢量的 `.pdf`)。
- **极简语义与纯净表达**：深度应用无干扰范式，即刻剔除不传递信息的余冗框架结构；依照实验断裂界点分离预先侵扰与后延脱溢期路径，匹配全图自动派生灰阶虚拟图例（Proxy Artist）。

## 4. 标准复现协议 (Standard Reproduction Protocol)

**执行序 1：全局计算配置界定**
预先检验最顶层控制域 `config.py` 的合规性：
1. 校准指向最终出版物图像贮存节点的绝对路由 `FINAL_OUTPUT_DIR`。
2. 配置目标数据组映射表 `EXPERIMENT_GROUPS`，声明关联的基础驱动路径与图形语义图例描述。
3. 对接底层数据尺度需求调配 `PARTICLE_RADIUS` 或相关的分析常数窗。

**执行序 2：并发数据缩减投影**
在保证系统计算池冗余的前提下投送提取任务包，并留意 `tqdm` 发送的心跳进度：
```bash
python 01_data_extractor.py
```

**执行序 3：专家在环质控校准 (Quality Control / Assurance)**
为了最大程度消除不规则几何体的微噪污染源，可启动图形化辅具系统：
```bash
python 01b_manual_corrector.py
```
- 若对当前帧计算出的物理极值持认可态度，轻点 `Enter` 键或关停对话框直接步入下一帧校验。
- 在遭遇明显散点引发模型伪判时，依循真实视界点取 **左端结构位点** 与对应 **右端结构位点**（可连续覆盖重置）。确认无误后按 `Enter` 键提交并触发局部重构。

**执行序 4：子系衍化诊断序列渲染**
批处理各单元形态结构与单相演变轨迹并落盘：
```bash
python 02_plot_single_diagnostic.py
```

**执行序 5：全局复合图论联合出装**
整合所有合理论证路径以生成跨组总汇动力系比对结果，抵达发布出版文件夹：
```bash
python 03_plot_multi_comparison.py
```

## 5. 常见问题与排错指南 (Troubleshooting & FAQ)

- **Q: `01` 系列底层提取器抛出内存异常警报 (Memory Error) 甚至造成宿主机内核无响应死锁？**
  - **A**: 并发引擎加载的热帧矩阵规模远远溢出了系统可用物理内存阈值。请先通过任务监控单元观测资源消耗峰值，若发现系统被挤压处于 Swap 分页过度负荷时，请尝试阻断当次任务，将 `01_data_extractor.py` 中 `concurrent.futures.ProcessPoolExecutor()` 的 `max_workers` 自定变量安全下调。如不具备进一步扩增主板驻留 RAM 的可行条件，则强制削减计算并发度。

- **Q: 在调用交互模块 `01b_manual_corrector.py` 的过程中窗口焦点游移导致侦听挂起或者标定拒绝接受输入行为？**
  - **A**: 此现象大概率归谬于 Matplotlib 多后端库的抢占冲突或本地视窗引擎渲染失误引发的异步阻塞。您可以尝试在 `01b` 代码顶部引源部分实施后端硬覆写对策，譬如追加 `import matplotlib; matplotlib.use('TkAgg')` 去锁定渲染层调度权限。

- **Q: 输出的全时段跨组演化比对曲线以及单帧峰位表现存在显著的不可忽视的剧烈高频假信号成分？**
  - **A**: 在特定断崖边缘与崩落模拟环境下散布着非连通游离态细流，导致全局极值或包线追踪遭遇物理形态与拟合算法的双非线性相交域误差干扰。建议即刻启运 `01b` 控制阀重置该阶干扰极小量；同时回溯 `config.py` 执行超参数调优行动：拉伸 `EXTRACT_SMOOTH_WINDOW` 或 `SMOOTHING_WINDOW` 特征长增加宏观滤波效准率。

---
**免责声明**
> 核心计算群严格杜绝回写污染机制，所有特征剥离工作及随附文件均局限于输出衍生池（如 CSV 及 PKL 表格格式存放）。离散元 `.dat` 输出模型群文件自始执行强只读访问，切除任何危及研究复现性的溯源灾变可能。经解耦系统升级优化，当前图像重建步骤不会倒逼重新唤醒高维空间阵列映射推演模块。

## 6. 如何引用 (How to Cite)

若此运算链路与处理逻辑为您的地球科学及地质工程学数值建模、或是相关连续介质非连续破坏力学论题的构架提供了确凿的分析证据解析价值与模型收敛助力，建议并且期待您能在最终投稿物（Manuscript）参考文献章节援引以下项目：

```bibtex
@article{zdem_salt_kinematics_pipeline,
  title     = {High-throughput Parameterization Pipeline for Kinematic Evolution in Salt Diapirism: A Deep Intervention Framework within Discrete Element Method},
  author    = {[To be updated], and [To be updated]},
  journal   = {[To be updated, e.g., Nature Geoscience / Geoscientific Model Development]},
  year      = {2026},
  volume    = {[To be updated]},
  number    = {[To be updated]},
  pages     = {[To be updated]},
  doi       = {10.xxxx/xxxx.xxxxx.xxxx}
}
```
*(注：占位引用标签会随同行评论阶段定稿最终出版物的 DOI 开源注册同步修订补齐以提供全域索引查询)*
