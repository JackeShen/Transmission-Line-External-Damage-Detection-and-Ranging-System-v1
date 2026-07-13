"""
LineUI 统一暗色主题样式表
基于 VSCode Dark+ 配色方案
"""

# ============================================================
# 配色常量
# ============================================================
COLORS = {
    # 背景
    'bg_primary': '#1e1e1e',
    'bg_secondary': '#252526',
    'bg_tertiary': '#2d2d30',
    'bg_hover': '#37373d',
    'bg_input': '#3c3c3c',

    # 边框
    'border_default': '#3e3e42',
    'border_focus': '#007acc',
    'border_accent': '#4ec9b0',

    # 文字
    'text_primary': '#cccccc',
    'text_secondary': '#858585',
    'text_bright': '#ffffff',
    'text_accent': '#4ec9b0',
    'text_warning': '#f48771',
    'text_highlight': '#569cd6',

    # 按钮
    'btn_primary': '#0e639c',
    'btn_primary_hover': '#1177bb',
    'btn_primary_pressed': '#005a9e',
    'btn_secondary': '#3e3e42',
    'btn_secondary_hover': '#4e4e4e',
    'btn_disabled_bg': '#2d2d30',
    'btn_disabled_text': '#555555',

    # 强调色
    'accent_cyan': '#4ec9b0',
    'accent_blue': '#569cd6',
    'accent_orange': '#ce9178',
    'accent_red': '#f44747',
    'accent_green': '#6a9955',
    'accent_yellow': '#dcdcaa',

    # 表格
    'table_bg': '#1e1e1e',
    'table_alt_bg': '#252526',
    'table_header_bg': '#2d2d30',
    'table_grid': '#3e3e42',
}

# ============================================================
# 全局样式表
# ============================================================
APP_STYLESHEET = f"""
/* ========== 全局 ========== */
QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}
QWidget {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: "Microsoft YaHei", "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}}

/* ========== 标签页 ========== */
QTabWidget::pane {{
    border: none;
    background: {COLORS['bg_primary']};
    padding: 0px;
}}
QTabBar::tab {{
    background: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
    padding: 10px 24px;
    margin-right: 1px;
    border: none;
    font-size: 13px;
    font-weight: 500;
    min-width: 100px;
}}
QTabBar::tab:selected {{
    background: {COLORS['bg_primary']};
    color: {COLORS['text_bright']};
    border-bottom: 2px solid {COLORS['border_focus']};
}}
QTabBar::tab:hover:!selected {{
    background: {COLORS['bg_hover']};
}}

/* ========== 分组框 ========== */
QGroupBox {{
    border: 1px solid {COLORS['border_default']};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 14px;
    font-size: 13px;
    font-weight: 600;
    color: {COLORS['text_primary']};
    background: {COLORS['bg_secondary']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 10px;
    color: {COLORS['text_accent']};
}}

/* ========== 按钮 ========== */
QPushButton {{
    background-color: {COLORS['btn_secondary']};
    color: {COLORS['text_bright']};
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {COLORS['btn_secondary_hover']};
    border-color: {COLORS['border_focus']};
}}
QPushButton:pressed {{
    background-color: {COLORS['bg_secondary']};
}}
QPushButton:disabled {{
    background-color: {COLORS['btn_disabled_bg']};
    color: {COLORS['btn_disabled_text']};
    border-color: {COLORS['border_default']};
}}

/* 主操作按钮（蓝色） */
QPushButton[cssClass="primaryBtn"] {{
    background-color: {COLORS['btn_primary']};
    color: white;
    border: none;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 20px;
    min-height: 38px;
}}
QPushButton[cssClass="primaryBtn"]:hover {{
    background-color: {COLORS['btn_primary_hover']};
}}
QPushButton[cssClass="primaryBtn"]:disabled {{
    background-color: {COLORS['btn_disabled_bg']};
    color: {COLORS['btn_disabled_text']};
}}

/* 小按钮 */
QPushButton[cssClass="smallBtn"] {{
    padding: 4px 12px;
    font-size: 12px;
    min-height: 26px;
}}

/* ========== 输入框 ========== */
QLineEdit {{
    background: {COLORS['bg_input']};
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
    color: {COLORS['text_bright']};
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: {COLORS['border_focus']};
}}
QLineEdit:focus {{
    border-color: {COLORS['border_focus']};
}}
QLineEdit::placeholder {{
    color: {COLORS['text_secondary']};
}}

/* ========== 复选框 ========== */
QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border_default']};
    border-radius: 3px;
    background: {COLORS['bg_primary']};
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['border_focus']};
}}
QCheckBox::indicator:checked {{
    background: {COLORS['border_focus']};
    border-color: {COLORS['border_focus']};
}}

/* ========== 下拉框 ========== */
QComboBox {{
    background: {COLORS['bg_input']};
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
    color: {COLORS['text_bright']};
    padding: 6px 10px;
    font-size: 13px;
    min-height: 20px;
}}
QComboBox:hover {{
    border-color: {COLORS['border_focus']};
}}
QComboBox QAbstractItemView {{
    background: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border_default']};
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['border_focus']};
}}

/* ========== 滑动条 ========== */
QSlider::groove:horizontal {{
    border: none;
    height: 4px;
    background: {COLORS['bg_tertiary']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['border_focus']};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS['btn_primary_hover']};
}}

/* ========== 表格 ========== */
QTableWidget {{
    background: {COLORS['table_bg']};
    alternate-background-color: {COLORS['table_alt_bg']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_default']};
    gridline-color: {COLORS['table_grid']};
    font-size: 13px;
    border-radius: 4px;
}}
QTableWidget::item {{
    padding: 6px 10px;
}}
QHeaderView::section {{
    background-color: {COLORS['table_header_bg']};
    color: {COLORS['text_primary']};
    padding: 8px 10px;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-bottom: 2px solid {COLORS['border_default']};
}}
QTableWidget::item:selected {{
    background-color: {COLORS['border_focus']};
}}

/* ========== 滚动条 ========== */
QScrollBar:vertical {{
    background: {COLORS['bg_primary']};
    width: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['bg_tertiary']};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['bg_hover']};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_primary']};
    height: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['bg_tertiary']};
    min-width: 20px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['bg_hover']};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ========== 滚动区域 ========== */
QScrollArea {{
    background: transparent;
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
}}

/* ========== 标签 ========== */
QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}

/* ========== 分割线 ========== */
QSplitter::handle {{
    background: {COLORS['border_default']};
    width: 1px;
}}

/* ========== 状态栏 ========== */
QStatusBar {{
    background: {COLORS['border_focus']};
    color: white;
    font-size: 12px;
    padding: 2px 10px;
    border: none;
}}

/* ========== 进度条 ========== */
QProgressBar {{
    background: {COLORS['bg_tertiary']};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    font-size: 11px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {COLORS['border_focus']};
    border-radius: 4px;
}}

/* ========== 工具提示 ========== */
QToolTip {{
    background: {COLORS['bg_secondary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_default']};
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 3px;
}}

/* ========== 数值微调框 ========== */
QDoubleSpinBox, QSpinBox {{
    background: {COLORS['bg_input']};
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
    color: {COLORS['text_bright']};
    padding: 5px 8px;
    font-size: 13px;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {COLORS['border_focus']};
}}
"""
