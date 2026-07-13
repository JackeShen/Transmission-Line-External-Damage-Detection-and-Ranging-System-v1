# 甘肃电网输电线外破检测测距系统 v2.0

基于机器视觉的输电线路外破目标检测与三维空间测距一体化平台。

## 功能概述

```
原始航拍图像
    ├─→ YOLO 模型 → 外破危险物检测 (边界框 + 类别)
    ├─→ Depth-Anything-V2 米制深度估计 → 稠密深度图 (单位: 米)
    └─→ 输电线数据 (JSON文件 或 DeepLab自动分割)
              ↓
        三维空间反投影 → 悬链线拟合 → KD-Tree最短距离
              ↓
        2D可视化: 检测框 + 危险物-导线连线 + 距离标注
```

## 目录结构

```
LineUI/
├── main.py                          # 程序入口
├── main_window.py                   # 主窗口 (系统风格)
├── core/
│   ├── engine.py                    # 测距引擎 (YOLO + Depth + 空间分析)
│   ├── project_manager.py           # 项目管理 (统一输入/输出目录)
│   ├── yolo_detector.py             # YOLO 模型封装
│   ├── deeplab_segmentor.py         # DeepLab 分割模型封装
│   └── spatial_analyzer.py          # 三维空间分析器
├── new_approach/                    # 三维测距算法
│   ├── pipeline.py                  # 10步测距管线
│   ├── distance.py                  # KD-Tree 距离计算 + 蒙特卡洛不确定性
│   ├── catenary.py                  # 悬链线拟合
│   ├── geometric_utils.py           # 相机模型 (针孔投影)
│   └── visualizer.py                # 2D/3D 可视化
├── Depth-Anything-V2-main/          # Depth-Anything-V2 (metric_depth)
│   └── metric_depth/run.py          # 命令行深度推理脚本
├── deeplab/                         # DeepLabV3+ 语义分割模块
├── widgets/
│   ├── step_panel.py                # 4步骤操作面板
│   └── image_viewer.py              # 自适应图像显示控件
├── workers/__init__.py              # QThread 后台线程 (6个)
├── theme/styles.py                  # VSCode Dark+ 暗色主题
└── utils/                           # mAP / IoU 评价指标 + CSV导出
```

## 环境配置

### 依赖
- Python 3.10+
- PyTorch ≥ 2.0
- PyQt5
- OpenCV, NumPy, SciPy, Matplotlib
- ultralytics (YOLO)
- timm (DeepLab backbone)

### Conda 环境 (推荐)
```bash
conda create -n goal python=3.10
conda activate goal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyqt5 opencv-python numpy scipy matplotlib ultralytics timm
```

## 快速开始

```bash
cd LineUI
python main.py
```

### 操作步骤

1. **设置项目目录** — 点击「📂 选择目录」或「🆕 新建项目」，所有输入输出将在此统一管理
2. **加载图像** — 选择航拍 RGB 图像 (JPG/PNG)
3. **加载 YOLO 模型** — 选择 `.pt` 权重文件 (如 `yolo11s-detection.pt`)
4. **加载深度模型** — 选择 Depth-Anything-V2 米制权重 (如 `output301.pth`)，选择对应 Encoder (vitb/vitl/vits)，设置 max_depth
5. **输电线数据** — 两种来源:
   - **JSON 文件**: 直接加载预标注的输电线分割 JSON
   - **DeepLab 检测**: 加载 DeepLab 模型，测距时自动分割输电线
6. **配置相机内参** — 修改 fx/fy/cx/cy (默认 DJI Zenmuse L1 参数)
7. **🚀 开始测距** — 自动执行: 检测 → 深度估计 → 三维测距 → 可视化

### 结果查看

- 📷 原始图像
- 🎯 YOLO 检测结果 (边界框)
- 🔮 深度图 (彩色热力图)
- 📐 测距可视化 (危险物-导线连线 + 距离标注)
- 📊 距离数据表 (ID / 类别 / 距离 / 安全等级)

## 模型文件

### YOLO 检测模型
- `project/model/yolo11s-detection.pt` — 外破危险物检测 (10类施工机械)
- `project/model/YOLOv11n_nest.pt` — 鸟巢检测

### Depth-Anything-V2 米制深度模型
- `checkpoints/depth_anything_v2_metric_vkitti_vitb.pth` — vitb, max_depth=80m
- `checkpoints/output301.pth` — 微调模型, vitb, max_depth=220m

### DeepLab 分割模型 (可选)
- `deeplab/best_epoch_weights.pth` — 输电线语义分割

## 命令行深度推理 (备选)

如果只需生成深度图，可用 `metric_depth/run.py`:

```bash
cd Depth-Anything-V2-main/metric_depth
python run.py \
    --encoder vitb \
    --max-depth 220 \
    --load-from ../checkpoints/output301.pth \
    --img-path /path/to/image.jpg \
    --outdir ./output \
    --save-numpy
```

输出: `*_raw_depth_meter.npy` (米制深度, 单位: 米)

## 输出文件

测距完成后在项目 `output/` 目录下生成:

```
output/
├── depth_metric.npy          # 米制深度图
├── detections.txt             # YOLO 检测结果 (归一化坐标)
├── power_line_seg.json        # DeepLab 分割的输电线 JSON
├── 2d_overlay.png             # 2D 可视化叠加图
├── distance_report.json       # 完整测距报告
└── distances_*.csv            # 导出距离表
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `QBasicTimer` 报错 | 已修复: 所有耗时操作均在 QThread 中执行 |
| 深度模型加载失败 | 检查 encoder 选择是否匹配权重文件 (vitb/vitl) |
| 测距报 `KeyError: 'bbox'` | 已修复: 自动转换归一化坐标 |
| matplotlib 子线程警告 | 已修复: 启动时设置 `matplotlib.use('Agg')` |
| NumPy 2.0 `ptp` 报错 | 已修复: `arr.ptp()` → `np.ptp(arr)` |
