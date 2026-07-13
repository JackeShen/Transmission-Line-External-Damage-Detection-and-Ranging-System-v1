#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lineUI/app.py  --  Power Line Corridor Distance Measurement System
PyQt5 GUI with two modes:
  1. Pipeline mode: image + depth + YOLO + wire JSON -> compute distances
  2. Click-measure mode: image + depth -> click two points -> 3D distance
"""

import sys, os, json, time, traceback
import numpy as np, cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QTextEdit, QSplitter, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QRadioButton, QButtonGroup,
    QScrollArea, QSizePolicy, QAbstractItemView, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from new_approach.geometric_utils import (
    CameraIntrinsics, pixel_to_camera, sample_depth_in_bbox, camera_to_pixel,
)
from new_approach.calibration import DepthCalibrator
from new_approach.distance import compute_distances, measure_two_points, DistanceResult

LEVEL_COLORS = {
    "SAFE": "#2ecc71", "WARNING": "#f39c12",
    "DANGER": "#e74c3c", "CRITICAL": "#8e44ad", "UNKNOWN": "#95a5a6",
}
LEVEL_ORDER = {"CRITICAL": 0, "DANGER": 1, "WARNING": 2, "SAFE": 3, "UNKNOWN": 4}

# ── 样式常量 ──────────────────────────────────────────────
STYLE_GROUP_TITLE = "font-weight:bold; font-size:12px; color:#2c3e50; padding: 4px 0;"
STYLE_FILE_LOADED = "color:#27ae60; font-weight:bold;"
STYLE_FILE_EMPTY  = "color:#bdc3c7;"
STYLE_RUN_BTN = (
    "QPushButton { font-size:14px; font-weight:bold; padding:10px; "
    "background:#2980b9; color:white; border-radius:6px; }"
    "QPushButton:hover { background:#3498db; }"
    "QPushButton:disabled { background:#95a5a6; }"
)
STYLE_BROWSE_BTN = (
    "QPushButton { padding:4px 10px; font-size:11px; }"
)
STYLE_LINEEDIT_RO = (
    "QLineEdit { background:#f8f9fa; border:1px solid #ddd; border-radius:3px; "
    "padding:3px 6px; font-size:11px; }"
)


# ═══════════════════════════════════════════════════════════
#  可复用组件
# ═══════════════════════════════════════════════════════════

def _make_file_row(label_text, parent, on_browse, placeholder="Not loaded"):
    """创建统一风格的文件输入行：[Label] [QLineEdit] [Browse]

    Returns:
        (row_layout, line_edit)
    """
    row = QHBoxLayout()
    row.setSpacing(6)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(75)
    lbl.setStyleSheet("font-weight:bold; font-size:11px;")
    row.addWidget(lbl)

    ed = QLineEdit()
    ed.setReadOnly(True)
    ed.setPlaceholderText(placeholder)
    ed.setStyleSheet(STYLE_LINEEDIT_RO)
    ed.setToolTip("")
    row.addWidget(ed, stretch=1)

    btn = QPushButton("...")
    btn.setFixedWidth(36)
    btn.setStyleSheet(STYLE_BROWSE_BTN)
    btn.setToolTip(f"Select {label_text.strip(':')} file")
    btn.clicked.connect(on_browse)
    row.addWidget(btn)

    return row, ed


def _set_file_loaded(edit, path):
    """设置 QLineEdit 为已加载状态：显示文件名 + tooltip 显示全路径"""
    name = os.path.basename(path) if path else ""
    edit.setText(name)
    edit.setToolTip(path)
    edit.setStyleSheet(
        "QLineEdit { background:#eafaf1; border:1px solid #27ae60; "
        "border-radius:3px; padding:3px 6px; font-size:11px; "
        "color:#1e8449; font-weight:bold; }"
    )
    edit.setCursorPosition(0)


# ═══════════════════════════════════════════════════════════
#  后台计算线程（不改）
# ═══════════════════════════════════════════════════════════

class ComputeWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, image, depth_map, danger_objects, wire_data,
                 intrinsics, scale_factor=None, calibration_objects=None,
                 depth_is_metric=False, danger_threshold=5.0, warning_threshold=10.0,
                 catenary_sample_spacing=0.1):
        super().__init__()
        self.image = image
        self.depth_map = depth_map
        self.danger_objects = danger_objects
        self.wire_data = wire_data
        self.intrinsics = intrinsics
        self.scale_factor = scale_factor
        self.calibration_objects = calibration_objects
        self.depth_is_metric = depth_is_metric
        self.danger_threshold = danger_threshold
        self.warning_threshold = warning_threshold
        self.catenary_sample_spacing = catenary_sample_spacing

    def run(self):
        try:
            self._do_compute()
        except Exception as e:
            self.error.emit(traceback.format_exc())

    def _do_compute(self):
        H, W = self.depth_map.shape
        self.intrinsics.width = W
        self.intrinsics.height = H

        self.progress.emit("Depth calibration...", 10)
        if self.calibration_objects and len(self.calibration_objects) >= 3:
            pairs = []
            for obj in self.calibration_objects:
                bbox, known = obj["bbox"], obj["distance"]
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                region = self.depth_map[y1:y2, x1:x2]
                valid = region[region > 0]
                if len(valid) == 0:
                    continue
                d = float(np.median(valid))
                pairs.append((d, known))
            if len(pairs) < 3:
                raise ValueError(f"Not enough calibration points: {len(pairs)} < 3")
            calibrator = DepthCalibrator(method="theil_sen", model="quadratic")
            calibrator.fit(pairs)
            converter = calibrator.converter
        elif self.scale_factor is not None:
            calibrator = DepthCalibrator(method="theil_sen", model="linear")
            fake = [(10.0, 10.0 * self.scale_factor),
                    (50.0, 50.0 * self.scale_factor),
                    (100.0, 100.0 * self.scale_factor)]
            calibrator.fit(fake)
            converter = calibrator.converter
        elif self.depth_is_metric:
            calibrator = None
            converter = lambda d: d
        else:
            raise ValueError("Select calibration mode: scale_factor / objects / metric")

        self.progress.emit("Converting depth to metric...", 20)
        depth_metric = np.zeros_like(self.depth_map, dtype=np.float64)
        mask = self.depth_map > 0
        depth_metric[mask] = converter(self.depth_map[mask])

        self.progress.emit("Projecting wires to 3D...", 30)
        for line in self.wire_data:
            pts = line.get("points", [])
            if len(pts) < 2:
                continue
            pts3d = []
            for x, y in pts:
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < W and 0 <= yi < H:
                    d = depth_metric[yi, xi]
                    if d > 0:
                        pts3d.append(pixel_to_camera(float(x), float(y), d, self.intrinsics))
            line["points_3d"] = pts3d

        self.progress.emit("Building wire KD-Trees...", 45)
        # 为每根导线建 KD-Tree（直接用3D标注点，不做悬链线拟合）
        from scipy.spatial import KDTree as KDTreeBuilder
        wire_3d_data = []
        for line in self.wire_data:
            pts = line.get("points_3d", [])
            if len(pts) >= 1:
                arr = np.array(pts)
                kd = KDTreeBuilder(arr)
                wire_3d_data.append({
                    "id": line.get("id", f"wire_{len(wire_3d_data)}"),
                    "points_3d": arr,
                    "kd_tree": kd,
                })
        if not wire_3d_data:
            raise ValueError("No valid wire 3D points - check wire JSON and depth map")

        self.progress.emit("Projecting hazards to 3D...", 60)
        danger_3d = []
        for oid, obj in self.danger_objects.items():
            bbox = obj["bbox"]
            md = sample_depth_in_bbox(depth_metric, bbox, percentile_low=10, percentile_high=90)
            if md <= 0:
                continue
            u = (bbox[0] + bbox[2]) / 2.0
            v = (bbox[1] + bbox[3]) / 2.0
            P = pixel_to_camera(u, v, md, self.intrinsics)
            danger_3d.append({
                "id": oid, "class_id": obj.get("class_id", -1),
                "bbox": bbox, "center_3d": P, "center_2d": (u, v),
            })
        if not danger_3d:
            raise ValueError("No valid depth values found for any hazard object")

        self.progress.emit("Computing distances...", 80)
        results = []
        for obj in danger_3d:
            P = obj["center_3d"]
            min_dist = float("inf")
            closest_wire_id = None
            closest_pt_3d = None

            for w in wire_3d_data:
                d, idx = w["kd_tree"].query(P)
                if d < min_dist:
                    min_dist = d
                    closest_wire_id = w["id"]
                    closest_pt_3d = w["points_3d"][idx]

            if closest_wire_id is None:
                continue

            # 安全等级
            if min_dist < self.danger_threshold:
                level = "DANGER"
            elif min_dist < self.warning_threshold:
                level = "WARNING"
            else:
                level = "SAFE"

            # 最近3D点 → 2D投影
            closest_2d = camera_to_pixel(closest_pt_3d, self.intrinsics)

            results.append(DistanceResult(
                object_id=obj["id"],
                class_id=obj["class_id"],
                bbox=tuple(obj["bbox"]),
                center_2d=obj["center_2d"],
                center_3d=P,
                distance=min_dist,
                closest_wire_id=closest_wire_id,
                closest_point_3d=closest_pt_3d,
                safety_level=level,
            ))
            # 附加2D最近点（供 _draw_overlay 画精准连线）
            results[-1].closest_point_2d = closest_2d

        self.progress.emit("Done!", 100)
        self.finished.emit(results, wire_3d_data)


# ═══════════════════════════════════════════════════════════
#  Matplotlib 画布（不改）
# ═══════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.tight_layout()

    def show_image(self, img_rgb):
        self.axes.clear()
        self.axes.imshow(img_rgb)
        self.axes.axis("off")
        self.fig.tight_layout()
        self.draw_idle()


# ═══════════════════════════════════════════════════════════
#  Tab 1: Pipeline（全流程测距）
# ═══════════════════════════════════════════════════════════

class PipelineTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.depth_map = None
        self.results = []
        self.wire_3d_data = []       # 导线3D点+KD-Tree（替代悬链线模型）
        self.wire_2d = None          # 导线2D标注点（用于画图）
        self._worker = None
        self._yolo_path = None
        self._wire_path = None
        self._init_ui()

    # ── UI 构建 ──────────────────────────────────────────

    def _init_ui(self):
        main_split = QSplitter(Qt.Horizontal)

        # ═══ 左侧面板 ═══
        left = QWidget()
        left.setMinimumWidth(340)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(10)

        # ── Input Files ──
        g1 = QGroupBox("Input Files")
        g1.setStyleSheet(STYLE_GROUP_TITLE)
        v1 = QVBoxLayout(g1)
        v1.setSpacing(8)

        row1, self.ed_img = _make_file_row("Image:", self, self._load_image)
        v1.addLayout(row1)
        row2, self.ed_dep = _make_file_row("Depth:", self, self._load_depth)
        v1.addLayout(row2)
        row3, self.ed_yolo = _make_file_row("Hazards:", self, self._load_yolo)
        v1.addLayout(row3)
        row4, self.ed_wire = _make_file_row("Wires:", self, self._load_wire_json)
        v1.addLayout(row4)
        ll.addWidget(g1)

        # ── Camera Intrinsics ──
        g2 = QGroupBox("Camera Intrinsics")
        g2.setStyleSheet(STYLE_GROUP_TITLE)
        f2 = QFormLayout(g2)
        f2.setSpacing(6)
        self.ed_fx = QLineEdit("3714.81"); self.ed_fy = QLineEdit("3714.81")
        self.ed_cx = QLineEdit("960");     self.ed_cy = QLineEdit("540")
        for w in [self.ed_fx, self.ed_fy, self.ed_cx, self.ed_cy]:
            w.setMaximumWidth(100)
        f2.addRow("fx:", self.ed_fx); f2.addRow("fy:", self.ed_fy)
        f2.addRow("cx:", self.ed_cx); f2.addRow("cy:", self.ed_cy)
        ll.addWidget(g2)

        # ── Depth Calibration ──
        g3 = QGroupBox("Depth Calibration")
        g3.setStyleSheet(STYLE_GROUP_TITLE)
        v3 = QVBoxLayout(g3)
        v3.setSpacing(6)

        self.rb_scale = QRadioButton("scale_factor")
        self.rb_obj   = QRadioButton("Calibration objects")
        self.rb_met   = QRadioButton("Already metric")
        self.rb_scale.setChecked(True)
        self._cg = QButtonGroup(self)
        self._cg.addButton(self.rb_scale, 0)
        self._cg.addButton(self.rb_obj, 1)
        self._cg.addButton(self.rb_met, 2)
        v3.addWidget(self.rb_scale)

        h_sf = QHBoxLayout()
        h_sf.addWidget(QLabel("Factor:"))
        self.ed_sf = QLineEdit("1.0")
        self.ed_sf.setMaximumWidth(80)
        h_sf.addWidget(self.ed_sf)
        h_sf.addStretch()
        v3.addLayout(h_sf)

        v3.addWidget(self.rb_obj)
        self.ed_calobj = QTextEdit()
        self.ed_calobj.setPlaceholderText("x1,y1,x2,y2,distance_m per line")
        self.ed_calobj.setMaximumHeight(70)
        self.ed_calobj.setStyleSheet("font-size:10px;")
        v3.addWidget(self.ed_calobj)

        v3.addWidget(self.rb_met)
        ll.addWidget(g3)

        # ── Thresholds ──
        g4 = QGroupBox("Safety Thresholds (m)")
        g4.setStyleSheet(STYLE_GROUP_TITLE)
        f4 = QFormLayout(g4)
        f4.setSpacing(4)
        self.ed_danger  = QLineEdit("5.0")
        self.ed_warning = QLineEdit("10.0")
        f4.addRow("DANGER <", self.ed_danger)
        f4.addRow("WARNING <", self.ed_warning)
        ll.addWidget(g4)

        # ── Run + Progress ──
        self.btn_run = QPushButton("Start Calculation")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet(STYLE_RUN_BTN)
        self.btn_run.clicked.connect(self._start)
        ll.addWidget(self.btn_run)

        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setMaximumHeight(18)
        ll.addWidget(self.pbar)

        self.st_lbl = QLabel("")
        self.st_lbl.setStyleSheet("color:#7f8c8d; font-size:11px;")
        ll.addWidget(self.st_lbl)

        ll.addStretch()

        # ═══ 右侧面板 ═══
        right = QSplitter(Qt.Vertical)

        self.canvas = MplCanvas(self)
        self.canvas.axes.text(0.5, 0.5, "Load image, depth, YOLO & wire JSON\nthen click Start",
                              transform=self.canvas.axes.transAxes, ha="center", va="center",
                              fontsize=13, color="#bdc3c7")
        self.canvas.axes.axis("off")
        self.canvas.draw_idle()
        right.addWidget(self.canvas)

        # 结果表格
        tc = QWidget()
        tl = QVBoxLayout(tc)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(4)

        hdr = QHBoxLayout()
        lbl_r = QLabel("Results")
        lbl_r.setStyleSheet("font-weight:bold; font-size:12px;")
        hdr.addWidget(lbl_r)
        hdr.addStretch()
        self.sum_lbl = QLabel("")
        self.sum_lbl.setStyleSheet("font-weight:bold; color:#2c3e50;")
        hdr.addWidget(self.sum_lbl)
        tl.addLayout(hdr)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Class", "Dist(m)", "Closest Wire", "Safety", "bbox", "3D"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        tl.addWidget(self.table)
        right.addWidget(tc)

        right.setSizes([520, 220])

        # ── 组合 ──
        main_split.addWidget(left)
        main_split.addWidget(right)
        main_split.setSizes([360, 880])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_split)

    # ── 文件加载 ──────────────────────────────────────────

    def _load_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All (*)")
        if not p: return
        img = cv2.imread(p)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot load: {p}")
            return
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _set_file_loaded(self.ed_img, p)
        H, W = self.image.shape[:2]
        self.ed_cx.setText(str(W // 2))
        self.ed_cy.setText(str(H // 2))
        self.canvas.show_image(self.image)

    def _load_depth(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Depth Map", "", "NumPy (*.npy);;All (*)")
        if not p: return
        self.depth_map = np.load(p)
        _set_file_loaded(self.ed_dep, p)

    def _load_yolo(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Labels", "", "Text (*.txt);;All (*)")
        if not p: return
        self._yolo_path = p
        _set_file_loaded(self.ed_yolo, p)

    def _load_wire_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Wire JSON", "", "JSON (*.json);;All (*)")
        if not p: return
        self._wire_path = p
        _set_file_loaded(self.ed_wire, p)

    # ── 数据解析（静态方法，不改逻辑）─────────────────────

    @staticmethod
    def _parse_yolo(path, iw, ih):
        objs = {}
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5: continue
                cid = int(parts[0]); cx, cy, w, h = map(float, parts[1:5])
                if cx <= 1.0 and cy <= 1.0 and w <= 1.0 and h <= 1.0:
                    x1 = int((cx - w/2) * iw); y1 = int((cy - h/2) * ih)
                    x2 = int((cx + w/2) * iw); y2 = int((cy + h/2) * ih)
                else:
                    x1, y1 = int(cx - w/2), int(cy - h/2)
                    x2, y2 = int(cx + w/2), int(cy + h/2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw-1, x2), min(ih-1, y2)
                objs[str(idx)] = {"class_id": cid, "bbox": [x1, y1, x2, y2]}
        return objs

    @staticmethod
    def _parse_wire_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list): return data
        if isinstance(data, dict):
            if "shapes" in data:
                return [{"id": s.get("label", "wire"),
                         "points": [(p[0], p[1]) for p in s["points"]]}
                        for s in data["shapes"]]
            if "power_lines" in data: return data["power_lines"]
            return [data]
        return []

    # ── 启动计算 ──────────────────────────────────────────

    def _start(self):
        if self.image is None or self.depth_map is None or \
           self._yolo_path is None or self._wire_path is None:
            QMessageBox.warning(self, "Notice",
                                "Please load all four files first:\n"
                                "Image / Depth / Hazards(.txt) / Wires(.json)")
            return

        H, W = self.image.shape[:2]
        try:
            fx = float(self.ed_fx.text()); fy = float(self.ed_fy.text())
            cx = float(self.ed_cx.text()); cy = float(self.ed_cy.text())
            intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                                          width=W, height=H)
        except ValueError:
            QMessageBox.critical(self, "Error", "Camera intrinsics must be numeric")
            return

        cm = self._cg.checkedId()
        sf = None; co = None; dm = False
        if cm == 2:
            dm = True
        elif cm == 1:
            raw = self.ed_calobj.toPlainText().strip()
            if not raw:
                QMessageBox.warning(self, "Notice", "Enter calibration objects")
                return
            co = []
            for line in raw.splitlines():
                parts = line.strip().split(",")
                if len(parts) < 5: continue
                try:
                    x1, y1, x2, y2 = map(int, parts[:4])
                    co.append({"bbox": [x1, y1, x2, y2],
                               "distance": float(parts[4])})
                except ValueError: continue
            if len(co) < 3:
                QMessageBox.warning(self, "Notice",
                                    f"Need >=3 objects, got {len(co)}")
                return
        else:
            try: sf = float(self.ed_sf.text())
            except ValueError:
                QMessageBox.critical(self, "Error",
                                     "scale_factor must be numeric")
                return

        try:
            dt = float(self.ed_danger.text())
            wt = float(self.ed_warning.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Thresholds must be numeric")
            return

        try:
            dangers = self._parse_yolo(self._yolo_path, W, H)
            wires = self._parse_wire_json(self._wire_path)
        except Exception as e:
            QMessageBox.critical(self, "Parse Error", traceback.format_exc())
            return

        self.wire_2d = wires  # 保存2D标注点，供 _draw_overlay 绘制导线

        if not dangers:
            QMessageBox.warning(self, "Notice", "No valid objects in YOLO file")
            return
        if not wires:
            QMessageBox.warning(self, "Notice", "No valid wires in JSON file")
            return

        self.btn_run.setEnabled(False)
        self.pbar.setVisible(True)
        self.pbar.setValue(0)
        self._worker = ComputeWorker(
            image=self.image, depth_map=self.depth_map.copy(),
            danger_objects=dangers, wire_data=wires, intrinsics=intrinsics,
            scale_factor=sf, calibration_objects=co, depth_is_metric=dm,
            danger_threshold=dt, warning_threshold=wt,
        )
        self._worker.progress.connect(
            lambda m, v: (self.st_lbl.setText(m), self.pbar.setValue(v)))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_err)
        self._worker.start()

    # ── 计算完成 ──────────────────────────────────────────

    def _on_done(self, results, wire_3d_data):
        self.results = results
        self.wire_3d_data = wire_3d_data
        self._fill_table()
        self._draw_overlay()
        self.btn_run.setEnabled(True)
        self.st_lbl.setText(
            f"Done — {len(results)} objects evaluated")

    def _on_err(self, msg):
        self.btn_run.setEnabled(True)
        self.pbar.setVisible(False)
        self.st_lbl.setText("Error — check console")
        QMessageBox.critical(self, "Error", msg)

    def _fill_table(self):
        sorted_r = sorted(self.results,
                          key=lambda r: LEVEL_ORDER.get(r.safety_level, 99))
        self.table.setRowCount(len(sorted_r))
        for row, r in enumerate(sorted_r):
            items = [
                QTableWidgetItem(r.object_id),
                QTableWidgetItem(str(r.class_id)),
                QTableWidgetItem(f"{r.distance:.2f}"),
                QTableWidgetItem(r.closest_wire_id),
                QTableWidgetItem(r.safety_level),
                QTableWidgetItem(
                    f"({r.bbox[0]},{r.bbox[1]},{r.bbox[2]},{r.bbox[3]})"),
                QTableWidgetItem(
                    f"({r.center_3d[0]:.1f},{r.center_3d[1]:.1f},{r.center_3d[2]:.1f})"),
            ]
            c = LEVEL_COLORS.get(r.safety_level, "#95a5a6")
            items[4].setBackground(QColor(c))
            items[4].setForeground(QColor("white"))
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)
        dng = sum(1 for r in self.results if r.safety_level == "DANGER")
        wrn = sum(1 for r in self.results if r.safety_level == "WARNING")
        saf = len(self.results) - dng - wrn
        self.sum_lbl.setText(
            f"Total: {len(self.results)}  |  "
            f"DANGER: {dng}  |  WARNING: {wrn}  |  SAFE: {saf}")

    # 导线2D配色（与 depth_to_3d.py 中 _WIRE_COLORS 一致）
    _WIRE_2D_COLORS = [
        "#e62e2e",   # 亮红
        "#ff8c00",   # 橙
        "#2ecc2e",   # 亮绿
        "#2980b9",   # 蓝
        "#e62ee6",   # 品红
    ]

    def _draw_overlay(self):
        if self.image is None: return
        ax = self.canvas.axes
        ax.clear()
        ax.imshow(self.image)

        # ── 1. 绘制导线标注线（最底层）──
        wire_legend_handles = []
        if self.wire_2d:
            for i, line in enumerate(self.wire_2d):
                pts = line.get("points", [])
                if len(pts) < 2:
                    continue
                wid = line.get("id", f"wire_{i}")
                c = self._WIRE_2D_COLORS[i % len(self._WIRE_2D_COLORS)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "-", color=c, linewidth=3, alpha=0.8,
                        zorder=1)
                wire_legend_handles.append(
                    mpatches.Patch(color=c, label=f"Wire: {wid}"))

        # ── 2. 绘制危险物检测框 + 到最近导线的距离连线 ──
        for r in self.results:
            x1, y1, x2, y2 = r.bbox
            level_color = LEVEL_COLORS.get(r.safety_level, "#95a5a6")
            u, v = r.center_2d

            # 使用 3D→2D 投影的精确最近点
            closest_2d = getattr(r, "closest_point_2d", None)

            # 绘制距离连线（先画，在框下面）
            if closest_2d is not None:
                wx, wy = closest_2d
                ax.plot([u, wx], [v, wy], "--", color="#f39c12",
                        linewidth=1.8, alpha=0.85, zorder=2)
                # 导线上最近点标注
                ax.plot(wx, wy, "o", color="#f39c12", markersize=6,
                        markeredgecolor="white", markeredgewidth=1,
                        zorder=3)

            # 检测框
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=level_color, linewidth=2.5,
                zorder=4)
            ax.add_patch(rect)

            # 距离标签（在框上方）
            ax.text(u, y1 - 10, f"{r.distance:.1f}m",
                    color="white", fontsize=8, fontweight="bold",
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=level_color, alpha=0.9))

            # 类别标签
            ax.text(x1 + 2, y1 + 2, f"#{r.class_id}",
                    color="white", fontsize=6, ha="left", va="top",
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="black", alpha=0.6))

        # ── 3. 图例 ──
        # 安全等级图例
        for lvl, lc in LEVEL_COLORS.items():
            if any(r.safety_level == lvl for r in self.results):
                wire_legend_handles.append(
                    mpatches.Patch(color=lc, label=lvl))
        if wire_legend_handles:
            ax.legend(handles=wire_legend_handles,
                      loc="upper right", fontsize=7,
                      framealpha=0.85, ncol=1)

        ax.set_title(
            f"Pipeline — {len(self.results)} objects, "
            f"{len(self.wire_2d) if self.wire_2d else 0} wires",
            fontsize=12, fontweight="bold")
        ax.axis("off")
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()



# ═══════════════════════════════════════════════════════════
#  Tab 2: 交互式两点测距
# ═══════════════════════════════════════════════════════════

class ClickMeasureTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.depth_map = None
        self.click_points = []
        self.measurements = []
        self._init_ui()

    def _init_ui(self):
        main_split = QSplitter(Qt.Horizontal)

        # ═══ 左侧面板 ═══
        left = QWidget()
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(10)

        # ── Input Files ──
        g1 = QGroupBox("Input Files")
        g1.setStyleSheet(STYLE_GROUP_TITLE)
        v1 = QVBoxLayout(g1)
        v1.setSpacing(8)

        row1, self.ed_img2 = _make_file_row("Image:", self, self._load_img)
        v1.addLayout(row1)
        row2, self.ed_dep2 = _make_file_row("Depth:", self, self._load_dep)
        v1.addLayout(row2)
        ll.addWidget(g1)

        # ── Camera Intrinsics ──
        g2 = QGroupBox("Camera Intrinsics")
        g2.setStyleSheet(STYLE_GROUP_TITLE)
        f2 = QFormLayout(g2)
        f2.setSpacing(6)
        self.ed_fx2 = QLineEdit("3714.81"); self.ed_fy2 = QLineEdit("3714.81")
        self.ed_cx2 = QLineEdit("960");     self.ed_cy2 = QLineEdit("540")
        for w in [self.ed_fx2, self.ed_fy2, self.ed_cx2, self.ed_cy2]:
            w.setMaximumWidth(100)
        f2.addRow("fx:", self.ed_fx2); f2.addRow("fy:", self.ed_fy2)
        f2.addRow("cx:", self.ed_cx2); f2.addRow("cy:", self.ed_cy2)
        ll.addWidget(g2)

        # ── Depth Calibration ──
        g3 = QGroupBox("Depth Calibration")
        g3.setStyleSheet(STYLE_GROUP_TITLE)
        v3 = QVBoxLayout(g3)
        v3.setSpacing(6)
        self.chk_met = QCheckBox("Already metric (no conversion)")
        v3.addWidget(self.chk_met)
        h_sf = QHBoxLayout()
        h_sf.addWidget(QLabel("scale_factor:"))
        self.ed_sf2 = QLineEdit("1.0")
        self.ed_sf2.setMaximumWidth(80)
        h_sf.addWidget(self.ed_sf2)
        h_sf.addStretch()
        v3.addLayout(h_sf)
        ll.addWidget(g3)

        # ── Actions ──
        g4 = QGroupBox("Actions")
        g4.setStyleSheet(STYLE_GROUP_TITLE)
        h4 = QHBoxLayout(g4)
        h4.setSpacing(8)
        bc = QPushButton("Clear Points")
        bu = QPushButton("Undo Last")
        bc.clicked.connect(self._clear)
        bu.clicked.connect(self._undo)
        bc.setStyleSheet(
            "QPushButton { padding:6px 14px; font-size:11px; }")
        bu.setStyleSheet(
            "QPushButton { padding:6px 14px; font-size:11px; }")
        h4.addWidget(bc)
        h4.addWidget(bu)
        ll.addWidget(g4)

        # ── Current Measurement ──
        g5 = QGroupBox("Current Measurement")
        g5.setStyleSheet(STYLE_GROUP_TITLE)
        v5 = QVBoxLayout(g5)
        self.cur_lbl = QLabel("Click two points on the image")
        self.cur_lbl.setWordWrap(True)
        self.cur_lbl.setStyleSheet(
            "font-size:12px; color:#2c3e50; padding:4px;")
        v5.addWidget(self.cur_lbl)
        ll.addWidget(g5)

        ll.addStretch()

        # ═══ 右侧面板 ═══
        right = QSplitter(Qt.Vertical)

        self.canvas = MplCanvas(self)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.axes.text(
            0.5, 0.5, "Load image & depth\nthen click two points to measure",
            transform=self.canvas.axes.transAxes,
            ha="center", va="center", fontsize=13, color="#bdc3c7")
        self.canvas.axes.axis("off")
        self.canvas.draw_idle()
        right.addWidget(self.canvas)

        # 测量历史
        tc = QWidget()
        tl2 = QVBoxLayout(tc)
        tl2.setContentsMargins(0, 0, 0, 0)
        tl2.setSpacing(4)

        hdr2 = QHBoxLayout()
        lbl_h = QLabel("Measurement History")
        lbl_h.setStyleSheet("font-weight:bold; font-size:12px;")
        hdr2.addWidget(lbl_h)
        hdr2.addStretch()
        self.hist_sum = QLabel("")
        self.hist_sum.setStyleSheet("color:#7f8c8d; font-size:11px;")
        hdr2.addWidget(self.hist_sum)
        tl2.addLayout(hdr2)

        self.ht = QTableWidget()
        self.ht.setColumnCount(6)
        self.ht.setHorizontalHeaderLabels(
            ["#", "P1(u,v)", "P2(u,v)", "Depth1(m)", "Depth2(m)",
             "3D Dist(m)"])
        self.ht.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ht.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ht.setAlternatingRowColors(True)
        self.ht.verticalHeader().setVisible(False)
        tl2.addWidget(self.ht)

        right.addWidget(tc)
        right.setSizes([520, 220])

        # ── 组合 ──
        main_split.addWidget(left)
        main_split.addWidget(right)
        main_split.setSizes([320, 880])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_split)

    # ── 文件加载 ──────────────────────────────────────────

    def _load_img(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All (*)")
        if not p: return
        img = cv2.imread(p)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot load: {p}")
            return
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _set_file_loaded(self.ed_img2, p)
        H, W = self.image.shape[:2]
        self.ed_cx2.setText(str(W // 2))
        self.ed_cy2.setText(str(H // 2))
        self.click_points.clear()
        self._redraw()

    def _load_dep(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Depth Map", "", "NumPy (*.npy);;All (*)")
        if not p: return
        self.depth_map = np.load(p)
        _set_file_loaded(self.ed_dep2, p)

    # ── 相机内参 ──────────────────────────────────────────

    def _get_intrinsics(self):
        try:
            fx = float(self.ed_fx2.text()); fy = float(self.ed_fy2.text())
            cx = float(self.ed_cx2.text()); cy = float(self.ed_cy2.text())
            H, W = self.image.shape[:2] if self.image is not None \
                else (1080, 1920)
            return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                                    width=W, height=H)
        except ValueError:
            return None

    # ── 点击交互 ──────────────────────────────────────────

    def _on_click(self, event):
        if self.image is None or self.depth_map is None:
            QMessageBox.warning(self, "Notice",
                                "Load image and depth first")
            return
        if event.inaxes != self.canvas.axes or event.xdata is None:
            return
        u, v = event.xdata, event.ydata
        H, W = self.image.shape[:2]
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))
        if len(self.click_points) >= 2:
            self.click_points.clear()
        self.click_points.append((u, v))
        if len(self.click_points) == 2:
            self._compute()
        else:
            self.cur_lbl.setText(
                f"P1: ({u:.0f}, {v:.0f}) — click second point")
            self._redraw()

    def _compute(self):
        p1, p2 = self.click_points
        intr = self._get_intrinsics()
        if intr is None:
            QMessageBox.critical(self, "Error", "Invalid intrinsics")
            return
        if self.chk_met.isChecked():
            conv = None
        else:
            try:
                sf = float(self.ed_sf2.text())
                conv = lambda d: d * sf
            except ValueError:
                QMessageBox.critical(self, "Error",
                                     "scale_factor must be numeric")
                return

        r = measure_two_points(p1, p2, self.depth_map, intr,
                               metric_converter=conv)
        if r is None:
            self.cur_lbl.setText(
                "Failed — invalid depth at selected point")
            self.click_points.clear()
            self._redraw()
            return

        self.measurements.append(r)
        self._fill_hist()
        self.cur_lbl.setText(
            f"P1({p1[0]:.0f},{p1[1]:.0f}) d={r.depth1_m:.1f}m  →  "
            f"P2({p2[0]:.0f},{p2[1]:.0f}) d={r.depth2_m:.1f}m\n"
            f"3D Distance: {r.distance_3d:.2f}m  |  "
            f"Image: {r.distance_2d_image:.0f}px")
        self._redraw()

    def _redraw(self):
        if self.image is None: return
        self.canvas.axes.clear()
        self.canvas.axes.imshow(self.image)
        for i, (u, v) in enumerate(self.click_points):
            c = "#3498db" if i == 0 else "#e74c3c"
            lb = "P1" if i == 0 else "P2"
            self.canvas.axes.plot(
                u, v, "o", color=c, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
            self.canvas.axes.text(
                u + 12, v - 8, lb, color="white",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=c))
        if len(self.click_points) == 2 and self.measurements:
            (u1, v1), (u2, v2) = self.click_points
            self.canvas.axes.plot(
                [u1, u2], [v1, v2], "--",
                color="#f39c12", linewidth=2)
            mu, mv = (u1 + u2) / 2, (v1 + v2) / 2
            self.canvas.axes.text(
                mu, mv - 10, f"{self.measurements[-1].distance_3d:.2f}m",
                color="white", fontsize=10, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f39c12"))
        self.canvas.axes.set_title(
            "Click Measure — Click any two points",
            fontsize=11, fontweight="bold")
        self.canvas.axes.axis("off")
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def _fill_hist(self):
        self.ht.setRowCount(len(self.measurements))
        for i, m in enumerate(self.measurements):
            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(
                    f"({m.point1_2d[0]:.0f},{m.point1_2d[1]:.0f})"),
                QTableWidgetItem(
                    f"({m.point2_2d[0]:.0f},{m.point2_2d[1]:.0f})"),
                QTableWidgetItem(f"{m.depth1_m:.1f}"),
                QTableWidgetItem(f"{m.depth2_m:.1f}"),
                QTableWidgetItem(f"{m.distance_3d:.2f}"),
            ]
            for col, item in enumerate(items):
                self.ht.setItem(i, col, item)
        self.ht.scrollToBottom()
        self.hist_sum.setText(f"{len(self.measurements)} measurements")

    def _clear(self):
        self.click_points.clear()
        self.cur_lbl.setText("Click two points on the image")
        self._redraw()

    def _undo(self):
        if self.click_points:
            self.click_points.pop()
            if not self.click_points:
                self.cur_lbl.setText("Click two points on the image")
            else:
                self.cur_lbl.setText(
                    f"P1: ({self.click_points[0][0]:.0f},"
                    f"{self.click_points[0][1]:.0f}) — click second point")
            self._redraw()


# ═══════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Power Line Corridor — Distance Measurement System v2.0")
        self.resize(1320, 860)
        self.tabs = QTabWidget()
        self.tabs.addTab(PipelineTab(), "Pipeline (Hazards→Wires)")
        self.tabs.addTab(ClickMeasureTab(), "Two-Point Measure")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage(
            "Ready — Load image and depth map to begin")


def main():
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 9))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
