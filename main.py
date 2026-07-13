"""
输电线路防外破测距系统
便携版 — 所有依赖均在 LineUI/ 内
"""
import sys
import os
# matplotlib 必须在导入 pyplot 前设为非交互模式（QThread 里用）
import matplotlib
matplotlib.use('Agg')

# LineUI/ 自身及其子模块
LINEUI_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_V2_DIR = os.path.join(LINEUI_DIR, 'Depth-Anything-V2-main')
METRIC_DEPTH_DIR = os.path.join(DEPTH_V2_DIR, 'metric_depth')
DEEPLAB_DIR = os.path.join(LINEUI_DIR, 'deeplab')
# 注意: insert(0) 后插入的反而在前面
# 最终顺序: DEEPLAB_DIR > METRIC_DEPTH_DIR > DEPTH_V2_DIR > LINEUI_DIR
for _path in [LINEUI_DIR, DEPTH_V2_DIR, METRIC_DEPTH_DIR, DEEPLAB_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("甘肃电网输电线外破检测测距系统")
    app.setApplicationVersion("2.0.0")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
