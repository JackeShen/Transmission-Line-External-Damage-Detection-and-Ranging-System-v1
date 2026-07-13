"""
后台工作线程 — 所有耗时操作放入 QThread
"""
import os
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np


# ============================================================
# 模型加载线程 (QThread, 避免 QBasicTimer 报错)
# ============================================================
class LoadYoloWorker(QThread):
    """后台加载 YOLO 模型"""
    finished = pyqtSignal(bool, int, str)  # ok, class_count, model_name
    error = pyqtSignal(str)

    def __init__(self, engine, model_path: str):
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def run(self):
        try:
            ok = self.engine.load_yolo(self.model_path)
            if ok:
                self.finished.emit(True, len(self.engine.class_names),
                                   self.model_path)
            else:
                self.error.emit("模型加载失败")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LoadDepthWorker(QThread):
    """后台加载 Depth-Anything-V2 模型"""
    finished = pyqtSignal(str, str)   # encoder, device
    error = pyqtSignal(str)

    def __init__(self, engine, model_path: str, encoder: str = 'vitl',
                 max_depth: float = None):
        super().__init__()
        self.engine = engine
        self.model_path = model_path
        self.encoder = encoder
        self.max_depth = max_depth

    def run(self):
        try:
            self.engine.load_depth_model(self.model_path, self.encoder,
                                         self.max_depth)
            self.finished.emit(self.engine.depth_encoder, self.engine.device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LoadDeepLabWorker(QThread):
    """后台加载 DeepLab 模型"""
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine, model_path: str):
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def run(self):
        try:
            self.engine.load_deeplab(self.model_path)
            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ============================================================
# YOLO 检测线程
# ============================================================
class DetectWorker(QThread):
    """YOLO目标检测"""
    finished = pyqtSignal(object, list)  # (annotated_image, detections)
    error = pyqtSignal(str)

    def __init__(self, engine, conf: float = 0.25, class_filter: list = None):
        super().__init__()
        self.engine = engine
        self.conf = conf
        self.class_filter = class_filter

    def run(self):
        try:
            img, dets = self.engine.run_detection(self.conf, self.class_filter)
            self.finished.emit(img, dets)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ============================================================
# 深度估计线程
# ============================================================
class DepthWorker(QThread):
    """Depth-Anything-V2 深度推理"""
    finished = pyqtSignal(np.ndarray)  # depth_array
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.progress.emit("正在进行深度估计...")
            depth = self.engine.run_depth_estimation()
            self.finished.emit(depth)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ============================================================
# 三维测距线程
# ============================================================
class RangingWorker(QThread):
    """三维空间测距 (新算法) — 含可选的 DeepLab 分割"""
    finished = pyqtSignal(object, object)  # (analyzer, results_dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, config: dict, deep_lab_mode: bool = False,
                 image_path: str = None):
        super().__init__()
        self.engine = engine
        self.config = config
        self.deeplab_mode = deep_lab_mode
        self.image_path = image_path

    def run(self):
        try:
            # DeepLab 模式: 先分割输电线 → JSON
            if self.deeplab_mode and self.image_path:
                self.progress.emit("DeepLab 检测输电线...")
                output_dir = self.config.get('output_dir', 'distance_results')
                json_path = os.path.join(output_dir, "power_line_seg.json")
                self.engine.run_deeplab_to_json(self.image_path, json_path)
                self.engine.set_power_line_json(json_path)

            self.progress.emit("执行米制深度估计...")
            if self.engine._depth_map is None:
                self.engine.run_depth_estimation()

            self.progress.emit("运行新算法三维测距...")
            output_dir = self.config.pop('output_dir', 'distance_results')
            results = self.engine.run_ranging(
                output_dir=output_dir,
                enable_smoothing=self.config.get('enable_smoothing', True),
                enable_uncertainty=False,
            )

            self.finished.emit(None, results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
