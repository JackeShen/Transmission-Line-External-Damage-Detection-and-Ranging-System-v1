"""
输电线路防外破测距系统 — 主窗口
单页面一体化设计，左侧步骤面板 + 右侧结果显示
"""
import os
import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QStatusBar,
    QSplitter, QMessageBox, QPushButton,
    QFileDialog
)
from PyQt5.QtGui import QColor

from theme.styles import APP_STYLESHEET, COLORS
from widgets import StepPanel, ImageViewer
from core import RangingEngine, ProjectManager
from workers import (
    DetectWorker, DepthWorker, RangingWorker,
    LoadYoloWorker, LoadDepthWorker, LoadDeepLabWorker
)


class MainWindow(QMainWindow):
    """输电线路防外破测距系统 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("甘肃电网输电线外破检测测距系统 v2.0")
        self.setGeometry(100, 100, 1500, 900)
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(APP_STYLESHEET)

        # 核心组件
        self.engine = RangingEngine()
        self.project = ProjectManager()

        # 状态
        self._image_path = None
        self._power_line_json = None
        self._depth_map = None
        self._detections = None
        self._detect_image = None

        self._init_ui()
        self._connect_signals()

    # ================================================================
    # UI 构建
    # ================================================================
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # 顶栏
        top_bar = QWidget()
        top_bar.setFixedHeight(44)
        top_bar.setStyleSheet(f"background: {COLORS['bg_secondary']}; border-bottom: 1px solid {COLORS['border_default']};")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(16)

        tb_layout.addStretch()
        app_title = QLabel("⚡ 甘肃电网输电线外破检测测距系统")
        app_title.setAlignment(Qt.AlignCenter)
        app_title.setStyleSheet(f"color: {COLORS['text_bright']}; font-size: 17px; font-weight: bold; border: none;")
        tb_layout.addWidget(app_title)
        tb_layout.addStretch()

        self.status_hint = QLabel("就绪")
        self.status_hint.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 12px; border: none;")
        tb_layout.addWidget(self.status_hint)

        # 主布局
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(top_bar)

        # 左右分栏
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # ---- 左侧: 步骤面板 ----
        self.step_panel = StepPanel()
        splitter.addWidget(self.step_panel)

        # ---- 右侧: 结果显示区 (2×2 网格 + 数据表) ----
        right = QWidget()
        right.setStyleSheet(f"background: {COLORS['bg_primary']};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(6)

        # 2x2 图像网格
        grid = QGridLayout()
        grid.setSpacing(6)

        self.orig_viewer = ImageViewer("等待加载图像...")
        grid.addWidget(self._panel_box("📷 原始图像", self.orig_viewer), 0, 0)

        self.detect_viewer = ImageViewer("等待检测...")
        grid.addWidget(self._panel_box("🎯 YOLO 检测", self.detect_viewer), 0, 1)

        self.depth_viewer = ImageViewer("等待深度估计...")
        grid.addWidget(self._panel_box("🔮 深度估计", self.depth_viewer), 1, 0)

        self.ranging_viewer = ImageViewer("等待测距...")
        grid.addWidget(self._panel_box("📐 测距可视化", self.ranging_viewer), 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        rl.addLayout(grid, 1)

        # 底部: 距离数据表 + 导出
        bottom = QWidget()
        bottom.setStyleSheet(f"background: {COLORS['bg_secondary']}; border-radius: 4px;")
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(4)

        tbl_header = QHBoxLayout()
        tbl_label = QLabel("📊 距离测量数据")
        tbl_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 13px; font-weight: bold; border: none;")
        tbl_header.addWidget(tbl_label)
        tbl_header.addStretch()
        self.export_btn = QPushButton("💾 导出 CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.setMaximumHeight(28)
        self.export_btn.clicked.connect(self._export_results)
        tbl_header.addWidget(self.export_btn)
        bl.addLayout(tbl_header)

        self.dist_table = QTableWidget()
        self.dist_table.setColumnCount(5)
        self.dist_table.setHorizontalHeaderLabels(["目标ID", "类别", "最短距离(m)", "最近导线", "状态"])
        self.dist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dist_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.dist_table.setFocusPolicy(Qt.NoFocus)
        self.dist_table.setShowGrid(False)
        self.dist_table.setAlternatingRowColors(True)
        self.dist_table.verticalHeader().setVisible(False)
        self.dist_table.verticalHeader().setDefaultSectionSize(28)
        self.dist_table.setMaximumHeight(160)
        self.dist_table.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['table_bg']};
                alternate-background-color: {COLORS['table_alt_bg']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_default']};
                font-size: 12px;
                border-radius: 3px;
            }}
            QHeaderView::section {{
                background: {COLORS['table_header_bg']};
                color: {COLORS['text_accent']};
                font-weight: bold;
                padding: 4px 8px;
                font-size: 12px;
                border: none;
                border-bottom: 2px solid {COLORS['border_focus']};
            }}
        """)
        bl.addWidget(self.dist_table)
        rl.addWidget(bottom)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        content_layout.addWidget(splitter, 1)

        # 状态栏
        status = QStatusBar()
        status.setStyleSheet(f"""
            QStatusBar {{ background: {COLORS['border_focus']}; border: none; padding: 2px; }}
            QStatusBar QLabel {{ color: white; font-size: 11px; }}
        """)
        self.status_label = QLabel("就绪")
        status.addWidget(self.status_label)
        self.setStatusBar(status)
        status.setVisible(True)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(content)

    def _panel_box(self, title: str, viewer: ImageViewer) -> QWidget:
        """工业风面板——紧凑标题栏 + 图像区"""
        box = QWidget()
        box.setStyleSheet(f"""
            QWidget {{ background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border_default']}; border-radius: 4px; }}
        """)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(f"""
            QWidget {{ background: {COLORS['bg_tertiary']}; border: none; border-radius: 4px 4px 0 0; border-bottom: 1px solid {COLORS['border_default']}; }}
        """)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 12px; font-weight: 600; border: none;")
        bl.addWidget(lbl)
        bl.addStretch()
        layout.addWidget(bar)

        layout.addWidget(viewer)
        return box

    # ================================================================
    # 信号连接
    # ================================================================
    def _connect_signals(self):
        sp = self.step_panel

        # 项目目录
        sp.working_dir_changed.connect(self._on_working_dir)

        # 步骤1: 图像
        sp.image_loaded.connect(self._on_image)

        # 步骤2: YOLO模型
        sp.yolo_model_loaded.connect(self._on_yolo_model)

        # 步骤3: 深度模型
        sp.depth_model_loaded.connect(self._on_depth_model)

        # 步骤4: 输电线数据
        sp.power_line_loaded.connect(self._on_power_line)
        sp.deeplab_model_loaded.connect(self._on_deeplab_model)

        # 开始测距
        sp.start_ranging.connect(self._on_start_ranging)

    # ================================================================
    # 业务逻辑
    # ================================================================
    def _on_working_dir(self, path: str):
        self.project.set_working_dir(path)
        self.status(f"项目目录: {path}")

    def _on_image(self, path: str):
        self._image_path = path
        # 导入到项目目录
        try:
            proj_path = self.project.import_image(path)
        except Exception:
            proj_path = path
        self.engine.set_image(proj_path)
        self.orig_viewer.show_file(proj_path)
        self.status(f"图像已加载: {os.path.basename(path)}")
        self._check_all_ready()

    def _on_yolo_model(self, path: str):
        self.status("正在加载YOLO模型...")
        self.step_panel.step2.set_status('active', '加载中...')
        self._yolo_load_worker = LoadYoloWorker(self.engine, path)
        self._yolo_load_worker.finished.connect(self._on_yolo_loaded)
        self._yolo_load_worker.error.connect(self._on_yolo_load_error)
        self._yolo_load_worker.start()

    def _on_yolo_loaded(self, ok: bool, n_classes: int, model_path: str):
        name = os.path.basename(model_path)
        self.step_panel.step2_ok(n_classes, name)
        self.status(f"YOLO模型: {name} ({n_classes}类)")
        self._check_all_ready()

    def _on_yolo_load_error(self, msg: str):
        self.step_panel.step2_error(msg)
        self.status(f"YOLO加载失败: {msg}")
        QMessageBox.critical(self, "YOLO模型加载失败", msg)

    def _on_depth_model(self, path: str, encoder: str):
        self.status("正在加载Depth-Anything-V2模型...")
        self.step_panel.step3.set_status('active', '加载中...')
        try:
            md = float(self.step_panel.max_depth_edit.text().strip() or 0)
        except ValueError:
            md = None
        self._depth_load_worker = LoadDepthWorker(self.engine, path, encoder, md)
        self._depth_load_worker.finished.connect(self._on_depth_loaded)
        self._depth_load_worker.error.connect(self._on_depth_load_error)
        self._depth_load_worker.start()

    def _on_depth_loaded(self, encoder: str, device: str):
        md = self.step_panel.max_depth_edit.text().strip()
        self.step_panel.step3_ok(encoder, f"{device} | max_depth={md}m")
        self.status(f"深度模型: {encoder} | max_depth={md}m | {device}")
        self._check_all_ready()

    def _on_depth_load_error(self, msg: str):
        self.step_panel.step3_error(msg)
        self.status(f"深度模型加载失败")
        QMessageBox.critical(self, "深度模型加载失败", msg)

    def _on_power_line(self, path: str):
        """输电线数据: JSON 路径 或 空字符串(DeepLab模式)"""
        self._power_line_json = path  # 空字符串 = DeepLab 模式, 测距时动态生成
        if path:
            try:
                proj_path = self.project.import_power_line_json(path)
            except Exception:
                proj_path = path
            self.engine.set_power_line_json(proj_path)
            self.status(f"输电线JSON: {os.path.basename(path)}")
        else:
            self.status("输电线数据: DeepLab 检测模式")
        self._check_all_ready()

    def _on_deeplab_model(self, path: str):
        """后台加载 DeepLab 模型 — 使用 QThread 避免 QBasicTimer 错误"""
        self._deeplab_load_worker = LoadDeepLabWorker(self.engine, path)
        self._deeplab_load_worker.finished.connect(self._on_deeplab_loaded)
        self._deeplab_load_worker.error.connect(self._on_deeplab_load_error)
        self._deeplab_load_worker.start()

    def _on_deeplab_loaded(self):
        self.status("DeepLab 模型加载成功")

    def _on_deeplab_load_error(self, msg: str):
        QMessageBox.critical(self, "DeepLab 加载失败", msg)

    def _check_all_ready(self):
        """检查是否所有步骤完成"""
        has_wire = bool(self._power_line_json) or (
            self._power_line_json == "" and self.engine.deeplab_loaded
        )
        ready = bool(
            self._image_path
            and self.engine.yolo_loaded
            and self.engine.depth_loaded
            and has_wire
        )
        self.step_panel.update_ready_state(ready)
        if ready:
            self.status_hint.setText("✅ 全部就绪，可以开始测距")

    def _on_start_ranging(self, config: dict):
        """开始完整测距管线"""
        self.status("开始测距管线...")
        self.step_panel.set_progress("准备中...", 0, 3)

        # 设置相机内参
        self.engine.set_camera_intrinsics(
            fx=config.get('fx'), fy=config.get('fy'),
            cx=config.get('cx'), cy=config.get('cy')
        )

        # 第一步: YOLO检测
        self._detect_worker = DetectWorker(self.engine)
        self._detect_worker.finished.connect(lambda img, dets: self._on_detect_done(img, dets, config))
        self._detect_worker.error.connect(self._on_detect_error)
        self._detect_worker.start()

    def _on_detect_done(self, img, detections, config: dict):
        self._detect_image = img
        self._detections = detections
        self.detect_viewer.show_cv(img)
        self.step_panel.set_progress(f"检测到 {len(detections)} 个目标 | 深度估计...", 1, 3)

        # 第二步: 米制深度估计
        self._depth_worker = DepthWorker(self.engine)
        self._depth_worker.finished.connect(lambda d: self._on_depth_done(d, config))
        self._depth_worker.error.connect(self._on_depth_error)
        self._depth_worker.start()

    def _on_depth_done(self, depth: np.ndarray, config: dict):
        self._depth_map = depth
        self.depth_viewer.show_depth_color(depth)
        self.step_panel.set_progress(f"米制深度: [{depth.min():.1f}, {depth.max():.1f}]m | 三维测距...", 2, 3)

        # 第三步: 新算法三维测距 (含可选 DeepLab 分割)
        config['output_dir'] = self.project.output_dir
        deeplab_mode = (self._power_line_json == "" and self.engine.deeplab_loaded)
        self._ranging_worker = RangingWorker(
            self.engine, config.copy(),
            deep_lab_mode=deeplab_mode,
            image_path=self._image_path
        )
        self._ranging_worker.progress.connect(lambda m: self.step_panel.set_progress(m))
        self._ranging_worker.finished.connect(self._on_ranging_done)
        self._ranging_worker.error.connect(self._on_ranging_error)
        self._ranging_worker.start()

    def _on_ranging_done(self, analyzer, results):
        self.step_panel.on_ranging_done()
        self.step_panel.set_progress("测距完成!", 3, 3)
        self.status("测距完成!")

        report = results.get('report')
        # 显示 2D 叠加可视化
        out_dir = self.project.output_dir
        for fname in ['2d_overlay.png']:
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                self.ranging_viewer.show_file(p)
                break

        # 用报告填充距离表
        res_list = report.results if report else []
        self.dist_table.setRowCount(len(res_list))
        for row, r in enumerate(res_list):
            self.dist_table.setItem(row, 0, QTableWidgetItem(r.object_id))
            self.dist_table.setItem(row, 1, QTableWidgetItem(str(r.class_id)))
            dist_item = QTableWidgetItem(f"{r.distance:.2f}")
            dist_item.setForeground(QColor(255, 100, 100) if r.distance < 5.0 else QColor(78, 201, 176))
            self.dist_table.setItem(row, 2, dist_item)
            self.dist_table.setItem(row, 3, QTableWidgetItem(r.closest_wire_id))
            status_item = QTableWidgetItem(r.safety_level)
            c = QColor(255, 100, 100) if r.safety_level in ('DANGER',) else QColor(220, 220, 100) if r.safety_level == 'WARNING' else QColor(106, 153, 85)
            status_item.setForeground(c)
            self.dist_table.setItem(row, 4, status_item)

        self.export_btn.setEnabled(True)

        danger_count = sum(1 for r in res_list if r.safety_level in ('DANGER',))
        self.status(f"测距完成 — {len(res_list)}个目标, {danger_count}个危险 (<5m)")
        QMessageBox.information(self, "测距完成",
            f"分析完成！\n总目标: {len(res_list)}\n危险 (<5m): {danger_count}")

    def _on_detect_error(self, msg: str):
        self.status(f"检测失败: {msg}")
        self.step_panel.on_ranging_error(msg)
        QMessageBox.critical(self, "错误", f"YOLO检测失败:\n{msg}")

    def _on_depth_error(self, msg: str):
        self.status(f"深度估计失败: {msg}")
        self.step_panel.on_ranging_error(msg)
        QMessageBox.critical(self, "错误", f"深度估计失败:\n{msg}")

    def _on_ranging_error(self, msg: str):
        self.status(f"测距失败: {msg}")
        self.step_panel.on_ranging_error(msg)
        QMessageBox.critical(self, "错误", f"测距分析失败:\n{msg}")

    def _export_results(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出测距结果", self.project.distances_csv_path(), "CSV (*.csv)")
        if path:
            from utils.export import export_distance_csv
            # 构建简单 results
            min_dists = []
            for row in range(self.dist_table.rowCount()):
                min_dists.append({
                    'danger_object_id': self.dist_table.item(row, 0).text(),
                    'class_id': self.dist_table.item(row, 1).text(),
                    'distance': float(self.dist_table.item(row, 2).text()),
                    'closest_line': self.dist_table.item(row, 3).text(),
                })
            export_distance_csv(path, {'min_distances': min_dists})
            self.status(f"结果已导出: {path}")

    def status(self, msg: str):
        self.status_label.setText(msg)


