"""new_approach — 基于深度图的输电走廊测距管线

与旧版 spatial_analyzer.py 完全独立，不修改原有代码。
核心改进：
  - 多点鲁棒标定（RANSAC / Theil-Sen）代替代替单点 scale_factor
  - 物理正确的悬链线拟合（在相机坐标系水平面内沿导线方向拟合高度）
  - 蒙特卡洛不确定性估计
  - 模块化、可测试的架构
"""

__version__ = "2.0.0"
