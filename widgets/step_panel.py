"""
步骤化操作面板 — 4步骤管线 + 参数配置 + 操作按钮
"""
import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QCheckBox, QProgressBar,
    QFrame, QScrollArea, QSizePolicy, QComboBox
)
from PyQt5.QtGui import QFont

from theme.styles import COLORS


class _StepCard(QFrame):
    """单个步骤卡片"""

    STATUS_PENDING = 'pending'
    STATUS_ACTIVE = 'active'
    STATUS_DONE = 'done'
    STATUS_ERROR = 'error'

    def __init__(self, number: int, title: str, action_text: str):
        super().__init__()
        self.number = number
        self.title = title
        self.action_text = action_text
        self._status = self.STATUS_PENDING
        self._detail = ""
        self._init_ui()

    def _init_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(14, 12, 14, 12)

        # 标题行
        header = QHBoxLayout()
        self.status_dot = QLabel("○")
        self.status_dot.setFixedWidth(24)
        self.status_dot.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 18px; font-weight: bold;")
        header.addWidget(self.status_dot)

        self.title_label = QLabel(f"步骤{self.number}: {self.title}")
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 500;")
        header.addWidget(self.title_label, 1)
        layout.addLayout(header)

        # 详情
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; padding-left: 30px;")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(False)
        layout.addWidget(self.detail_label)

        self._refresh_style()

    def set_status(self, status: str, detail: str = ""):
        self._status = status
        self._detail = detail
        if detail:
            self.detail_label.setText(detail)
            self.detail_label.setVisible(True)
        self._refresh_style()

    def _refresh_style(self):
        colors = {
            self.STATUS_PENDING: (COLORS['text_secondary'], COLORS['bg_secondary'], COLORS['border_default']),
            self.STATUS_ACTIVE:   (COLORS['border_focus'], COLORS['bg_secondary'], COLORS['border_focus']),
            self.STATUS_DONE:     (COLORS['text_accent'], COLORS['bg_secondary'], COLORS['text_accent']),
            self.STATUS_ERROR:    (COLORS['accent_red'], COLORS['bg_secondary'], COLORS['accent_red']),
        }
        dot_map = {
            self.STATUS_PENDING: "○",
            self.STATUS_ACTIVE: "◉",
            self.STATUS_DONE: "●",
            self.STATUS_ERROR: "✕",
        }
        dot_color, bg, border = colors.get(self._status, colors[self.STATUS_PENDING])
        self.status_dot.setText(dot_map.get(self._status, "○"))
        self.status_dot.setStyleSheet(f"color: {dot_color}; font-size: 18px; font-weight: bold;")
        self.setStyleSheet(f"""
            _StepCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 6px;
            }}
        """)


# ============================================================
# 步骤面板
# ============================================================
class StepPanel(QWidget):
    """左侧步骤化操作面板"""

    # 信号
    image_loaded = pyqtSignal(str)
    yolo_model_loaded = pyqtSignal(str)
    depth_model_loaded = pyqtSignal(str, str)   # model_path, encoder
    power_line_loaded = pyqtSignal(str)         # JSON 路径 或 空字符串(DeepLab模式)
    deeplab_model_loaded = pyqtSignal(str)      # DeepLab 模型路径
    start_ranging = pyqtSignal(dict)            # config dict
    working_dir_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("StepPanel")
        self.setFixedWidth(340)
        self.setStyleSheet(f"#StepPanel {{ background: {COLORS['bg_secondary']}; }}")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {COLORS['bg_secondary']}; border: none; }}")

        inner = QWidget()
        inner.setStyleSheet(f"background: {COLORS['bg_secondary']};")
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ====== 标题 ======
        title = QLabel("⚡ 测距工作流")
        title.setStyleSheet(f"color: {COLORS['text_bright']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color: {COLORS['border_default']};")
        layout.addWidget(divider)

        # ====== 项目目录 ======
        proj_group = QGroupBox("📁 项目目录")
        pl = QVBoxLayout(proj_group)
        pl.setSpacing(8)
        pl.setContentsMargins(10, 16, 10, 10)

        self.dir_label = QLabel("未设置")
        self.dir_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.dir_label.setWordWrap(True)
        pl.addWidget(self.dir_label)

        dir_btn_row = QHBoxLayout()
        self.set_dir_btn = QPushButton("📂 选择目录")
        self.set_dir_btn.clicked.connect(self._choose_dir)
        dir_btn_row.addWidget(self.set_dir_btn)
        self.new_dir_btn = QPushButton("🆕 新建项目")
        self.new_dir_btn.clicked.connect(self._new_project)
        dir_btn_row.addWidget(self.new_dir_btn)
        pl.addLayout(dir_btn_row)

        layout.addWidget(proj_group)

        # ====== 步骤卡片 ======
        steps_group = QGroupBox("🔧 操作步骤")
        sl = QVBoxLayout(steps_group)
        sl.setSpacing(10)
        sl.setContentsMargins(10, 16, 10, 10)

        # 步骤1: 加载图像
        self.step1 = _StepCard(1, "加载图像", "选择图像")
        sl.addWidget(self.step1)
        self._load_image_btn = QPushButton("🖼️ 选择原始图像")
        self._load_image_btn.clicked.connect(self._load_image)
        sl.addWidget(self._load_image_btn)

        # 步骤2: YOLO模型
        self.step2 = _StepCard(2, "加载YOLO模型", "选择模型")
        sl.addWidget(self.step2)
        self._load_yolo_btn = QPushButton("🎯 加载YOLO模型")
        self._load_yolo_btn.clicked.connect(self._load_yolo)
        sl.addWidget(self._load_yolo_btn)

        # 步骤3: 深度模型
        self.step3 = _StepCard(3, "加载深度估计模型", "选择模型")
        sl.addWidget(self.step3)
        # encoder 选择
        enc_row = QHBoxLayout()
        enc_row.addWidget(QLabel("Encoder:"))
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(['vitb', 'vitl', 'vits', 'vitg'])
        self.encoder_combo.setCurrentText('vitb')
        enc_row.addWidget(self.encoder_combo)
        sl.addLayout(enc_row)
        self._load_depth_btn = QPushButton("🔮 加载Depth-Anything-V2")
        self._load_depth_btn.clicked.connect(self._load_depth)
        sl.addWidget(self._load_depth_btn)

        # 步骤4: 输电线数据 (JSON 或 DeepLab)
        self.step4 = _StepCard(4, "输电线数据来源", "选择来源")
        sl.addWidget(self.step4)

        # 来源选择
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("来源:"))
        self.wire_source_combo = QComboBox()
        self.wire_source_combo.addItems(["📄 JSON文件", "🔬 DeepLab检测"])
        self.wire_source_combo.currentIndexChanged.connect(self._on_wire_source_changed)
        src_row.addWidget(self.wire_source_combo)
        sl.addLayout(src_row)

        # JSON 文件选择
        self._wire_json_widget = QWidget()
        wjl = QHBoxLayout(self._wire_json_widget)
        wjl.setContentsMargins(0, 0, 0, 0)
        self._load_pl_btn = QPushButton("📄 选择输电线JSON")
        self._load_pl_btn.clicked.connect(self._load_power_line)
        wjl.addWidget(self._load_pl_btn)
        sl.addWidget(self._wire_json_widget)

        # DeepLab 模型选择
        self._wire_deeplab_widget = QWidget()
        wdl = QVBoxLayout(self._wire_deeplab_widget)
        wdl.setContentsMargins(0, 0, 0, 0)
        wdl.setSpacing(6)
        self.deeplab_path_label = QLabel("未加载模型")
        self.deeplab_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        wdl.addWidget(self.deeplab_path_label)
        self._load_deeplab_btn = QPushButton("🔬 加载DeepLab模型")
        self._load_deeplab_btn.clicked.connect(self._load_deeplab_model)
        wdl.addWidget(self._load_deeplab_btn)
        self._wire_deeplab_widget.hide()
        sl.addWidget(self._wire_deeplab_widget)

        layout.addWidget(steps_group)

        # ====== 参数配置 ======
        param_group = QGroupBox("⚙️ 相机内参 (米制深度, 无需标定)")
        pml = QVBoxLayout(param_group)
        pml.setSpacing(10)
        pml.setContentsMargins(10, 16, 10, 10)

        # 提示
        hint = QLabel("米制深度模型 (metric_depth)\nmax_depth 控制最大输出距离")
        hint.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 11px; padding: 4px;")
        pml.addWidget(hint)

        # max_depth
        md_row = QHBoxLayout()
        md_row.addWidget(QLabel("max_depth(m):"))
        self.max_depth_edit = QLineEdit("80")
        self.max_depth_edit.setPlaceholderText("80 (vkitti) 或 20 (hypersim)")
        md_row.addWidget(self.max_depth_edit)
        pml.addLayout(md_row)

        cam_grid = QVBoxLayout()
        cam_grid.setSpacing(6)
        defaults = [("fx", "3714.81"), ("fy", "3714.81"), ("cx", "2640.0"), ("cy", "1978.0")]
        for name, default_val in defaults:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            edit = QLineEdit()
            edit.setPlaceholderText(default_val)
            edit.setText(default_val)
            row.addWidget(edit)
            setattr(self, f"cam_{name}_edit", edit)
            cam_grid.addLayout(row)
        pml.addLayout(cam_grid)

        layout.addWidget(param_group)

        # ====== 操作按钮 ======
        self.start_btn = QPushButton("🚀 开始测距")
        self.start_btn.setProperty("cssClass", "primaryBtn")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        layout.addStretch()

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ================================================================
    # 操作
    # ================================================================
    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择项目工作目录")
        if d:
            self.dir_label.setText(d)
            self.dir_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 11px; font-weight: bold;")
            self.working_dir_changed.emit(d)

    def _new_project(self):
        d = QFileDialog.getExistingDirectory(self, "选择父目录")
        if d:
            import os
            proj = os.path.join(d, "ranging_project")
            os.makedirs(proj, exist_ok=True)
            self.dir_label.setText(proj)
            self.dir_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 11px; font-weight: bold;")
            self.working_dir_changed.emit(proj)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择原始图像", "", "图像 (*.jpg *.jpeg *.png *.bmp);;所有 (*.*)")
        if path:
            self.step1.set_status(_StepCard.STATUS_DONE, os.path.basename(path))
            self.image_loaded.emit(path)

    def _load_yolo(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "模型 (*.pt *.onnx);;所有 (*.*)")
        if path:
            self.step2.set_status(_StepCard.STATUS_ACTIVE, "加载中...")
            self.yolo_model_loaded.emit(path)

    def _load_depth(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择Depth-Anything-V2权重", "", "模型 (*.pth *.pt);;所有 (*.*)")
        if path:
            self.step3.set_status(_StepCard.STATUS_ACTIVE, "加载中...")
            encoder = self.encoder_combo.currentText()
            self.depth_model_loaded.emit(path, encoder)

    def _on_wire_source_changed(self, idx: int):
        """切换输电线数据来源"""
        if idx == 0:  # JSON
            self._wire_json_widget.show()
            self._wire_deeplab_widget.hide()
            self.step4.set_status(_StepCard.STATUS_PENDING, "请选择JSON文件")
        else:  # DeepLab
            self._wire_json_widget.hide()
            self._wire_deeplab_widget.show()
            self.step4.set_status(_StepCard.STATUS_PENDING, "请加载DeepLab模型")

    def _load_power_line(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输电线分割JSON", "", "JSON (*.json);;所有 (*.*)")
        if path:
            self.step4.set_status(_StepCard.STATUS_DONE, f"JSON: {os.path.basename(path)}")
            self.power_line_loaded.emit(path)

    def _load_deeplab_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择DeepLab权重", "", "模型 (*.pth *.pt);;所有 (*.*)")
        if path:
            self.step4.set_status(_StepCard.STATUS_ACTIVE, "加载中...")
            self.deeplab_path_label.setText(f"✓ {os.path.basename(path)}")
            self.deeplab_path_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 11px; font-weight: bold;")
            self.deeplab_model_loaded.emit(path)
            # 触发就绪检查 — power_line_loaded 传空字符串表示使用DeepLab
            self.power_line_loaded.emit("")
            self.step4.set_status(_StepCard.STATUS_DONE, "DeepLab模型就绪")

    def wire_source_is_json(self) -> bool:
        return self.wire_source_combo.currentIndex() == 0

    def _start(self):
        config = {}
        # 相机内参
        for k in ['fx', 'fy', 'cx', 'cy']:
            v = getattr(self, f"cam_{k}_edit").text().strip()
            if v:
                try:
                    config[k] = float(v)
                except ValueError:
                    pass
        self.start_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.start_ranging.emit(config)

    # ================================================================
    # 外部调用
    # ================================================================
    def step2_ok(self, class_count: int, model_name: str):
        self.step2.set_status(_StepCard.STATUS_DONE, f"{model_name} | {class_count}个类别")

    def step2_error(self, msg: str):
        self.step2.set_status(_StepCard.STATUS_ERROR, msg)

    def step3_ok(self, encoder: str, device: str):
        self.step3.set_status(_StepCard.STATUS_DONE, f"{encoder} | 设备: {device}")

    def step3_error(self, msg: str):
        self.step3.set_status(_StepCard.STATUS_ERROR, msg)

    def set_progress(self, text: str, value: int = None, maximum: int = None):
        if maximum is not None:
            self.progress.setRange(0, maximum)
        if value is not None:
            self.progress.setValue(value)
        self.progress.setFormat(f"  {text}")

    def on_ranging_done(self):
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)

    def on_ranging_error(self, msg: str):
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)

    def update_ready_state(self, all_ready: bool):
        self.start_btn.setEnabled(all_ready)
