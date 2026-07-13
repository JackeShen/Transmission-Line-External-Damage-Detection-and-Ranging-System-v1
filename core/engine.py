"""
测距引擎 — 使用新算法 (new_approach)
米制深度估计 → 无需标定, npy 直接为米单位
"""
import os
import cv2
import numpy as np
import torch

from core.yolo_detector import YOLODetector
from depth_anything_v2.dpt import DepthAnythingV2
from new_approach.pipeline import Pipeline, PipelineConfig
from new_approach.geometric_utils import CameraIntrinsics
from core.deeplab_segmentor import DeepLabSegmentor


class RangingEngine:
    """测距引擎: YOLO检测 + Depth-Anything-V2 米制深度 + 新测距算法"""

    MODEL_CONFIGS = {
        'vits':  {'encoder': 'vits',  'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb':  {'encoder': 'vitb',  'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl':  {'encoder': 'vitl',  'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg':  {'encoder': 'vitg',  'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    def __init__(self):
        self.yolo_detector: YOLODetector = None
        self.depth_model = None
        self.depth_encoder = 'vitl'
        self.device = self._get_device()
        self._image_path: str = None
        self._power_line_json: str = None
        self._depth_map: np.ndarray = None
        self._detections: list = None
        # 相机内参（默认值，可在UI中修改）
        self._fx = 3714.81
        self._fy = 3714.81
        self._cx = 2640.0
        self._cy = 1978.0
        self._last_report = None

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    # ========== 模型加载 ==========

    def load_yolo(self, model_path: str) -> bool:
        detector = YOLODetector()
        if not detector.load_model(model_path):
            return False
        self.yolo_detector = detector
        return True

    def load_depth_model(self, model_path: str, encoder: str = 'vitl',
                         max_depth: float = None) -> bool:
        """加载米制深度模型 (使用 metric_depth 修改版 DPT)

        encoder / max_depth 可手动指定, 留空则从文件名自动推断:
          ...metric_vkitti_vitb.pth → encoder='vitb', max_depth=80
          ...metric_hypersim_vitl.pth → encoder='vitl', max_depth=20
        """
        fname = os.path.basename(model_path).lower()

        if encoder == 'vitl':  # 默认值, 尝试自动检测
            for enc in ['vits', 'vitb', 'vitl', 'vitg']:
                if enc in fname:
                    encoder = enc
                    break

        if max_depth is None:
            if 'vkitti' in fname:
                max_depth = 80.0
            elif 'hypersim' in fname:
                max_depth = 20.0
            else:
                max_depth = 20.0

        cfg = self.MODEL_CONFIGS[encoder]
        model = DepthAnythingV2(**{**cfg, 'max_depth': max_depth})

        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location='cpu')

        model.load_state_dict(state)
        model = model.to(self.device).eval()
        self.depth_model = model
        self.depth_encoder = encoder
        return True

    # ========== DeepLab 分割模型 ==========

    _deeplab_segmentor: DeepLabSegmentor = None

    def load_deeplab(self, model_path: str, num_classes: int = 2,
                     backbone: str = "convnextv2") -> bool:
        seg = DeepLabSegmentor()
        if not seg.load_model(model_path, num_classes=num_classes, backbone=backbone):
            return False
        self._deeplab_segmentor = seg
        return True

    @property
    def deeplab_loaded(self) -> bool:
        return self._deeplab_segmentor is not None and self._deeplab_segmentor.is_loaded()

    def run_deeplab_to_json(self, image_path: str, output_json: str) -> str:
        """运行 DeepLab 分割, 将输电线 mask 转为 pipeline 所需的 JSON 格式"""
        import json
        from PIL import Image

        seg = self._deeplab_segmentor
        if seg is None:
            raise RuntimeError("请先加载 DeepLab 模型")

        # 获取分割 mask
        mask = seg.predict_mask(image_path)
        h, w = mask.shape[:2]

        # 提取输电线(非背景)的所有像素轮廓
        wire_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(wire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 每个轮廓作为一条输电线
        lines = []
        for i, cnt in enumerate(contours):
            if len(cnt) < 5:
                continue
            # 简化轮廓点（每隔几个点取一个）
            pts = cnt.squeeze(1).tolist()
            if len(pts) > 200:
                step = max(1, len(pts) // 200)
                pts = pts[::step]
            lines.append({"id": f"wire_{i}", "points": pts})

        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(lines, f, ensure_ascii=False, indent=2)

        print(f"DeepLab 分割完成: {len(lines)} 条输电线 → {output_json}")
        return output_json

    @property
    def yolo_loaded(self) -> bool:
        return self.yolo_detector is not None and self.yolo_detector.is_loaded()

    @property
    def depth_loaded(self) -> bool:
        return self.depth_model is not None

    @property
    def class_names(self) -> list:
        return self.yolo_detector.get_class_names() if self.yolo_detector else []

    # ========== 相机参数 ==========

    def set_camera_intrinsics(self, fx=None, fy=None, cx=None, cy=None):
        if fx is not None: self._fx = fx
        if fy is not None: self._fy = fy
        if cx is not None: self._cx = cx
        if cy is not None: self._cy = cy

    # ========== 数据设置 ==========

    def set_image(self, image_path: str):
        self._image_path = image_path

    def set_power_line_json(self, json_path: str):
        self._power_line_json = json_path

    # ========== 执行 ==========

    def run_detection(self, conf_threshold: float = 0.25,
                      class_filter: list = None) -> tuple:
        if not self.yolo_loaded or not self._image_path:
            raise RuntimeError("请先加载YOLO模型和图像")
        annotated, detections = self.yolo_detector.detect(
            self._image_path, conf_threshold, class_filter
        )
        self._detections = detections
        return annotated, detections

    def run_depth_estimation(self, save_npy_path: str = None) -> np.ndarray:
        """米制深度估计 — 输出单位为米"""
        if not self.depth_loaded or not self._image_path:
            raise RuntimeError("请先加载深度模型和图像")
        img = cv2.imread(self._image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self._image_path}")
        depth = self.depth_model.infer_image(img)
        self._depth_map = depth
        if save_npy_path:
            np.save(save_npy_path, depth)
        return depth

    def run_ranging(self, output_dir: str = "distance_results",
                    enable_smoothing: bool = True,
                    enable_uncertainty: bool = False) -> dict:
        """
        执行三维测距 (米制深度, 无需标定)
        返回 {'report': DistanceReport, 'pipeline': Pipeline}
        """
        if self._depth_map is None:
            raise RuntimeError("请先运行深度估计")
        if not self._power_line_json:
            raise RuntimeError("请先加载输电线JSON文件")
        if not self._image_path:
            raise RuntimeError("请先加载图像")

        # 保存深度图为 npy（新算法需要文件路径）
        npy_path = os.path.join(output_dir, "depth_metric.npy")
        os.makedirs(output_dir, exist_ok=True)
        np.save(npy_path, self._depth_map)

        # 生成 YOLO 标注文件（新算法需要 txt 文件）
        anno_path = self._write_yolo_annotations(output_dir)

        # 构建相机内参
        img = cv2.imread(self._image_path)
        h, w = img.shape[:2]
        intrinsics = CameraIntrinsics(
            fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
            width=w, height=h
        )

        # 新算法 Pipeline (scale_factor=1.0 因为深度已是米制)
        config = PipelineConfig(
            image_path=self._image_path,
            depth_map_path=npy_path,
            annotation_path=anno_path,
            segmentation_path=self._power_line_json,
            camera_intrinsics=intrinsics,
            calibration_objects=[],         # 不需要标定物
            scale_factor=1.0,               # 米制深度: 恒等变换
            output_dir=output_dir,
            danger_threshold=5.0,
            warning_threshold=10.0,
            enable_uncertainty=enable_uncertainty,
            enable_2d_viz=True,
            enable_3d_viz=False,
            enable_combined_viz=False,
        )

        pipeline = Pipeline(config)
        report = pipeline.run()
        self._last_report = report
        return {'report': report, 'pipeline': pipeline}

    def _write_yolo_annotations(self, output_dir: str) -> str:
        """将 YOLO 检测结果写为 txt 文件（归一化坐标）"""
        anno_path = os.path.join(output_dir, "detections.txt")
        img = cv2.imread(self._image_path)
        h, w = img.shape[:2]
        with open(anno_path, 'w') as f:
            for det in (self._detections or []):
                x1, y1, x2, y2 = det['bbox']
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        return anno_path

    def is_ready(self) -> bool:
        return (self.yolo_loaded and self.depth_loaded
                and self._image_path and self._power_line_json)
