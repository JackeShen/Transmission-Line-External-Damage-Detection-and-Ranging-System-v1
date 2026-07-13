"""
图像显示控件 — 支持自适应缩放、彩色深度图
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel

from theme.styles import COLORS


class ImageViewer(QLabel):
    """可自适应的图像显示控件"""

    def __init__(self, placeholder: str = "等待加载..."):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(200)
        self.setScaledContents(False)
        self._placeholder = placeholder
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText(self._placeholder)
        self.setStyleSheet(f"""
            QLabel {{
                background: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border_default']};
                color: {COLORS['text_secondary']};
                font-size: 15px;
                border-radius: 6px;
            }}
        """)

    def _set_active_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                background: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)

    def show_file(self, path: str):
        """显示图像文件"""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._set_placeholder()
            return
        self._set_active_style()
        self._scale_and_set(pixmap)

    def show_cv(self, img: np.ndarray):
        """显示 OpenCV 图像 (BGR)"""
        if img is None:
            self._set_placeholder()
            return
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            # 灰度图
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self._set_active_style()
        self._scale_and_set(pixmap)

    def show_depth_color(self, depth: np.ndarray):
        """彩色映射显示深度图"""
        d = depth - depth.min()
        d = d / (d.max() + 1e-6)
        d8 = (d * 255).astype(np.uint8)
        colored = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
        self.show_cv(colored)

    def _scale_and_set(self, pixmap: QPixmap):
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            size = pixmap.size()
        scaled = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
