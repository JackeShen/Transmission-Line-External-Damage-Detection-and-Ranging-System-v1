"""pipeline ¡ª main orchestration

Ties all modules together into a complete measurement pipeline.
"""

import os, json, time
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .geometric_utils import (
    CameraIntrinsics, pixel_to_camera,
    sample_depth_in_bbox,
)
from .calibration import DepthCalibrator, auto_calibrate
from .catenary import fit_all_wires, CatenaryModel
from .distance import (
    compute_distances, estimate_uncertainty,
    DistanceResult, DistanceReport,
)
from .visualizer import (
    draw_2d_overlay, draw_3d_point_cloud, draw_combined,
    print_report,
)


@dataclass
class PipelineConfig:
    image_path: str = ""
    depth_map_path: str = ""
    annotation_path: str = ""
    segmentation_path: str = ""
    camera_intrinsics: CameraIntrinsics = None
    calibration_objects: List[Dict] = field(default_factory=list)
    output_dir: str = "new_approach_results"
    scale_factor: Optional[float] = None
    danger_threshold: float = 5.0
    warning_threshold: float = 10.0
    catenary_sample_spacing: float = 0.1
    enable_uncertainty: bool = True
    mc_samples: int = 500
    pixel_noise: float = 2.0
    enable_2d_viz: bool = True
    enable_3d_viz: bool = True
    enable_combined_viz: bool = True


class Pipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.intrinsics = config.camera_intrinsics
        self.image = None
        self.depth_map = None
        self.depth_metric = None
        self.danger_objects = {}
        self.power_line_data = []
        self.calibrator = None
        self.wire_models = []
        self.results = []
        self.uncertainties = []
        self.report = None

    def run(self):
        t0 = time.time()
        print("=" * 60)
        print("  NEW APPROACH ¡ª Distance Measurement Pipeline")
        print("=" * 60)
        self._step1_load_inputs()
        self._step2_calibrate()
        self._step3_convert_depth()
        self._step4_project_wires()
        self._step5_fit_catenaries()
        self._step6_project_dangers()
        self._step7_compute_distances()
        if self.config.enable_uncertainty:
            self._step8_estimate_uncertainty()
        self._step9_visualize()
        self._step10_save_report()
        elapsed = time.time() - t0
        print(f"\n  Pipeline done in {elapsed:.1f}s")
        print(f"  Output: {os.path.abspath(self.config.output_dir)}")
        return self.report

    def _step1_load_inputs(self):
        print("\n[1/10] Loading inputs...")
        cfg = self.config
        img = cv2.imread(cfg.image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {cfg.image_path}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = self.image.shape[:2]
        self.intrinsics.width = W
        self.intrinsics.height = H
        print(f"  Image: {W}x{H}")
        self.depth_map = np.load(cfg.depth_map_path)
        print(f"  Depth: {self.depth_map.shape}, range=[{self.depth_map.min():.2f}, {self.depth_map.max():.2f}]")
        self.danger_objects = self._load_yolo(cfg.annotation_path)
        print(f"  Dangers: {len(self.danger_objects)}")
        # 转换归一化坐标为像素坐标
        detect_and_convert_bbox(self, W, H)
        self.power_line_data = self._load_wire_json(cfg.segmentation_path)
        print(f"  Wires: {len(self.power_line_data)} groups")

    def _step2_calibrate(self):
        print("\n[2/10] Depth calibration...")
        cfg = self.config
        if cfg.calibration_objects:
            self.calibrator = auto_calibrate(self.depth_map, cfg.calibration_objects)
            r = self.calibrator.result
            print(f"  Model: {r.model_type}, Method: {r.method}")
            print(f"  RMSE: {r.rmse:.3f}m, MAE: {r.mae:.3f}m, R2: {r.r_squared:.4f}")
            print(f"  Points: {r.n_points}, Depth range: [{r.depth_range[0]:.2f}, {r.depth_range[1]:.2f}]")
        elif cfg.scale_factor is not None:
            print(f"  Using manual scale_factor={cfg.scale_factor}")
            self.calibrator = DepthCalibrator(method="theil_sen", model="linear")
            fake = [(10.0, 10.0 * cfg.scale_factor), (50.0, 50.0 * cfg.scale_factor), (100.0, 100.0 * cfg.scale_factor)]
            self.calibrator.fit(fake)
        else:
            raise ValueError("Need calibration_objects or scale_factor")

    def _step3_convert_depth(self):
        print("\n[3/10] Metric depth conversion...")
        converter = self.calibrator.converter
        self.depth_metric = np.zeros_like(self.depth_map, dtype=np.float64)
        mask = self.depth_map > 0
        self.depth_metric[mask] = converter(self.depth_map[mask])
        vals = self.depth_metric[mask]
        print(f"  Metric range: [{vals.min():.2f}, {vals.max():.2f}] m")

    def _step4_project_wires(self):
        print("\n[4/10] Wire 2D->3D projection...")
        for line in self.power_line_data:
            pts = line.get("points", [])
            if len(pts) < 2:
                continue
            pts3d = []
            for x, y in pts:
                xi, yi = int(round(float(x))), int(round(float(y)))
                W, H = self.intrinsics.width, self.intrinsics.height
                if 0 <= xi < W and 0 <= yi < H:
                    d = self.depth_metric[yi, xi]
                    if d > 0:
                        pts3d.append(pixel_to_camera(float(x), float(y), d, self.intrinsics))
            line["points_3d"] = pts3d
            print(f"  {line.get('id','?')}: {len(pts3d)}/{len(pts)} valid 3D")

    def _step5_fit_catenaries(self):
        print("\n[5/10] Catenary fitting...")
        groups = {}
        for line in self.power_line_data:
            pts = line.get("points_3d", [])
            if len(pts) >= 5:
                wid = line.get("id", f"wire_{len(groups)}")
                groups[wid] = pts
        self.wire_models = fit_all_wires(groups, sample_spacing=self.config.catenary_sample_spacing)
        print(f"  Fitted: {len(self.wire_models)}/{len(self.power_line_data)} wires")

    def _step6_project_dangers(self):
        print("\n[6/10] Danger 2D->3D projection...")
        for oid, obj in self.danger_objects.items():
            bbox = obj["bbox"]
            md = sample_depth_in_bbox(self.depth_metric, bbox, percentile_low=10, percentile_high=90)
            if md <= 0:
                print(f"  {oid}: depth sampling failed, skip")
                continue
            u = (bbox[0] + bbox[2]) / 2.0
            v = (bbox[1] + bbox[3]) / 2.0
            P = pixel_to_camera(u, v, md, self.intrinsics)
            obj["center_3d"] = P
            obj["center_2d"] = (u, v)
            obj["depth_metric"] = md
            print(f"  {oid}: 3D=({P[0]:.1f}, {P[1]:.1f}, {P[2]:.1f})")

    def _step7_compute_distances(self):
        print("\n[7/10] Distance computation...")
        d3d = []
        for oid, obj in self.danger_objects.items():
            if obj.get("center_3d") is not None:
                d3d.append(dict(
                    id=oid,
                    class_id=obj.get("class_id", -1),
                    bbox=obj["bbox"],
                    center_3d=obj["center_3d"],
                    center_2d=obj["center_2d"],
                ))
        self.results = compute_distances(d3d, self.wire_models,
            danger_threshold=self.config.danger_threshold,
            warning_threshold=self.config.warning_threshold)
        for r in self.results:
            print(f"  {r.object_id}: {r.distance:.2f}m -> {r.closest_wire_id} [{r.safety_level}]")

    def _step8_estimate_uncertainty(self):
        print(f"\n[8/10] Monte Carlo uncertainty ({self.config.mc_samples} samples)...")
        d3d = []
        for oid, obj in self.danger_objects.items():
            if obj.get("center_3d") is not None:
                d3d.append(dict(
                    id=oid,
                    class_id=obj.get("class_id", -1),
                    bbox=obj["bbox"],
                    center_3d=obj["center_3d"],
                    center_2d=obj["center_2d"],
                ))
        self.uncertainties = estimate_uncertainty(d3d, self.wire_models,
            self.depth_map, self.intrinsics, self.calibrator.converter,
            calibration_rmse=self.calibrator.result.rmse,
            n_samples=self.config.mc_samples,
            pixel_noise=self.config.pixel_noise)
        umap = {u["object_id"]: u for u in self.uncertainties}
        for r in self.results:
            u = umap.get(r.object_id)
            if u:
                r.uncertainty_95 = u["ci_95_half_width"]
                print(f"  {r.object_id}: {r.distance:.2f}m 95%CI=[{u['ci_95_low']:.2f}, {u['ci_95_high']:.2f}]m")

    def _step9_visualize(self):
        print("\n[9/10] Visualization...")
        cfg = self.config
        out = cfg.output_dir
        os.makedirs(out, exist_ok=True)
        w2d = []
        for line in self.power_line_data:
            if "points" in line:
                w2d.append((line["points"], line.get("id", "?")))
        if cfg.enable_2d_viz:
            draw_2d_overlay(self.image, self.results, w2d,
                os.path.join(out, "2d_overlay.png"),
                show_uncertainty=cfg.enable_uncertainty)
        if cfg.enable_3d_viz:
            draw_3d_point_cloud(self.results, self.wire_models,
                os.path.join(out, "3d_point_cloud.png"))
        if cfg.enable_combined_viz:
            draw_combined(cfg.image_path, self.results, self.wire_models, w2d,
                os.path.join(out, "combined_report.png"))

    def _step10_save_report(self):
        print("\n[10/10] Save report...")
        cal_rmse = self.calibrator.result.rmse if self.calibrator else 0.0
        self.report = DistanceReport(
            results=self.results,
            wire_models=self.wire_models,
            calibration_rmse=cal_rmse,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        out = self.config.output_dir
        jp = os.path.join(out, "distance_report.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  JSON: {jp}")
        print_report(self.results, self.uncertainties, cal_rmse)

    @staticmethod
    def _load_yolo(path):
        objs = {}
        with open(path, "r") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            if cx <= 1.0 and cy <= 1.0 and w <= 1.0 and h <= 1.0:
                objs[str(idx)] = dict(class_id=cid, bbox_norm=[cx, cy, w, h])
            else:
                objs[str(idx)] = dict(class_id=cid, bbox=[int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)])
        return objs

    @staticmethod
    def _load_wire_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "shapes" in data:
                return [dict(id=s.get("label","wire"), points=[(p[0],p[1]) for p in s["points"]]) for s in data["shapes"]]
            if "power_lines" in data:
                return data["power_lines"]
            return [data]
        return []


def detect_and_convert_bbox(pipeline, iw, ih):
    for oid, obj in pipeline.danger_objects.items():
        if "bbox_norm" in obj:
            cx, cy, w, h = obj["bbox_norm"]
            x1 = int((cx - w/2) * iw)
            y1 = int((cy - h/2) * ih)
            x2 = int((cx + w/2) * iw)
            y2 = int((cy + h/2) * ih)
            obj["bbox"] = [max(0,x1), max(0,y1), min(iw-1,x2), min(ih-1,y2)]
            del obj["bbox_norm"]


# Patch interactive_click_measure onto Pipeline so you can call
#   pipeline.interactive_measure() after pipeline.run()
def _pipeline_interactive_measure(self, enable_uncertainty: bool = False,
                                  output_path: Optional[str] = None):
    """Run the interactive click-to-measure tool on this pipeline's data.

    Convenience wrapper around :func:`interactive_click_measure`.

    Args:
        enable_uncertainty: 是否启用蒙特卡洛不确定性（较慢）
        output_path: 保存标注结果的图像路径

    Returns:
        [TwoPointMeasurement, ...]
    """
    from .visualizer import interactive_click_measure

    cal_rmse = self.calibrator.result.rmse if self.calibrator else 0.0
    # depth_metric is already in meters; no converter needed
    return interactive_click_measure(
        image=self.image,
        depth_map=self.depth_metric,
        intrinsics=self.intrinsics,
        metric_converter=None,   # depth_metric 已是米制
        calibration_rmse=cal_rmse,
        enable_uncertainty=enable_uncertainty,
        output_path=output_path,
    )


Pipeline.interactive_measure = _pipeline_interactive_measure
