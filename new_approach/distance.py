"""distance — 距离计算与不确定性估计

提供：
  - 基于 KD-Tree 的点到导线最近距离计算
  - 点→线段距离（导线采样点间插值）
  - 蒙特卡洛不确定性传播
  - 安全等级分类
  - 两点交互式测距（点击图像任意两点 → 3D距离）
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial import KDTree

from .catenary import CatenaryModel
from .geometric_utils import CameraIntrinsics, pixel_to_camera


@dataclass
class DistanceResult:
    """单个危险物到导线的距离结果"""
    object_id: str
    class_id: int
    bbox: Tuple[int, int, int, int]
    center_2d: Tuple[float, float]
    center_3d: np.ndarray            # 相机坐标系 3D 位置
    distance: float                  # 到最近导线的距离 (m)
    closest_wire_id: str             # 最近导线的 ID
    closest_point_3d: np.ndarray     # 导线上最近点（相机坐标）
    # 不确定性
    uncertainty_95: Optional[float] = None  # 95% 置信区间半宽 (m)
    # 安全等级
    safety_level: str = "UNKNOWN"    # SAFE | WARNING | DANGER | CRITICAL


@dataclass
class DistanceReport:
    """完整测距报告"""
    results: List[DistanceResult]
    wire_models: List[CatenaryModel]
    calibration_rmse: float          # 标定 RMSE (m)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "results": [
                {
                    "object_id": r.object_id,
                    "class_id": r.class_id,
                    "bbox": list(r.bbox),
                    "center_3d": r.center_3d.tolist(),
                    "distance_m": round(r.distance, 3),
                    "closest_wire": r.closest_wire_id,
                    "closest_point_3d": r.closest_point_3d.tolist(),
                    "uncertainty_95": round(r.uncertainty_95, 3) if r.uncertainty_95 else None,
                    "safety_level": r.safety_level,
                }
                for r in self.results
            ],
            "wire_count": len(self.wire_models),
            "calibration_rmse": round(self.calibration_rmse, 3),
        }


def compute_distances(danger_objects_3d: List[Dict],
                      wire_models: List[CatenaryModel],
                      danger_threshold: float = 5.0,
                      warning_threshold: float = 10.0,
                      ) -> List[DistanceResult]:
    """计算每个危险物到所有导线的最短距离

    Args:
        danger_objects_3d: [{'id': str, 'class_id': int, 'bbox': [...],
                              'center_3d': np.array([X,Y,Z]), 'center_2d': (u,v)}, ...]
        wire_models: 悬链线模型列表
        danger_threshold: 危险距离阈值 (m)
        warning_threshold: 警告距离阈值 (m)

    Returns:
        DistanceResult 列表
    """
    if not wire_models:
        return []

    results = []
    for obj in danger_objects_3d:
        P = obj['center_3d']
        min_dist = float('inf')
        closest_wire = None
        closest_pt = None

        for wm in wire_models:
            # 先查 KD-Tree
            dist_kd, idx = wm.kd_tree.query(P)

            # 如果 KD-Tree 最近点旁边有点，做线段插值精化
            samples = wm.samples_3d
            if 0 < idx < len(samples) - 1:
                # 点到线段距离（在 KD 最近点前后各取一个邻居做插值）
                dist_seg, proj = _point_to_segment(P, samples[idx], samples[idx + 1])
                dist_seg2, _ = _point_to_segment(P, samples[idx - 1], samples[idx])
                if dist_seg < dist_kd:
                    dist_kd = dist_seg
                if dist_seg2 < dist_kd:
                    dist_kd = dist_seg2

            if dist_kd < min_dist:
                min_dist = dist_kd
                closest_wire = wm.wire_id
                # 重新查一次取精确最近点
                d, i = wm.kd_tree.query(P)
                closest_pt = wm.samples_3d[i]

        if closest_wire is None:
            continue

        # 安全等级
        if min_dist < danger_threshold:
            level = "DANGER"
        elif min_dist < warning_threshold:
            level = "WARNING"
        else:
            level = "SAFE"

        results.append(DistanceResult(
            object_id=obj['id'],
            class_id=obj['class_id'],
            bbox=tuple(obj['bbox']),
            center_2d=obj['center_2d'],
            center_3d=P,
            distance=min_dist,
            closest_wire_id=closest_wire,
            closest_point_3d=closest_pt,
            safety_level=level,
        ))

    return results


def estimate_uncertainty(danger_objects_3d: List[Dict],
                         wire_models: List[CatenaryModel],
                         depth_map: np.ndarray,
                         intrinsics: CameraIntrinsics,
                         metric_converter,
                         calibration_rmse: float,
                         n_samples: int = 500,
                         pixel_noise: float = 2.0,
                         depth_noise_relative: float = 0.02,
                         ) -> List[Dict]:
    """蒙特卡洛不确定性估计

    对每个危险物：
      1. 在其 2D 中心位置加高斯噪声（±pixel_noise px）
      2. 在深度值上加噪声（±depth_noise_relative × depth + calibration_rmse）
      3. 重新投影到 3D 并计算距离
      4. 统计距离分布，输出中位数和 95% 置信区间

    Args:
        danger_objects_3d: 危险物列表
        wire_models: 导线模型
        depth_map: 原始深度图（相对值）
        intrinsics: 相机内参
        metric_converter: 深度→米制转换函数
        calibration_rmse: 标定 RMSE (m)
        n_samples: 蒙特卡洛采样数
        pixel_noise: 像素噪声标准差 (px)
        depth_noise_relative: 深度值相对噪声比例

    Returns:
        [{'object_id': str, 'median': float, 'std': float, 'ci_95_low': float,
          'ci_95_high': float, 'samples': [...]}, ...]
    """
    H, W = depth_map.shape
    uncertainties = []

    for obj in danger_objects_3d:
        u, v = obj['center_2d']
        ui, vi = int(round(u)), int(round(v))
        # 获取该点的原始深度值
        if 0 <= vi < H and 0 <= ui < W:
            raw_depth = depth_map[vi, ui]
        else:
            raw_depth = 0.0

        distances = []
        for _ in range(n_samples):
            # 像素位置加噪声
            u_noisy = u + np.random.normal(0, pixel_noise)
            v_noisy = v + np.random.normal(0, pixel_noise)

            # 深度值加噪声
            depth_noisy = raw_depth * (1.0 + np.random.normal(0, depth_noise_relative))
            depth_noisy = max(depth_noisy, 1e-6)
            metric_d = metric_converter(depth_noisy)
            metric_d += np.random.normal(0, calibration_rmse)
            metric_d = max(metric_d, 0.1)

            # 3D 投影
            P_noisy = pixel_to_camera(u_noisy, v_noisy, metric_d, intrinsics)

            # 计算距离
            min_d = float('inf')
            for wm in wire_models:
                d, _ = wm.kd_tree.query(P_noisy)
                if d < min_d:
                    min_d = d
            distances.append(min_d)

        distances = np.array(distances)
        median = np.median(distances)
        std = np.std(distances)
        ci_low = np.percentile(distances, 2.5)
        ci_high = np.percentile(distances, 97.5)

        uncertainties.append({
            'object_id': obj['id'],
            'median': median,
            'std': std,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'ci_95_half_width': (ci_high - ci_low) / 2.0,
        })

    return uncertainties


@dataclass
class TwoPointMeasurement:
    """两点测距结果"""
    point1_2d: Tuple[float, float]
    point2_2d: Tuple[float, float]
    point1_3d: np.ndarray          # 相机坐标系 [Xc, Yc, Zc]
    point2_3d: np.ndarray
    depth1_m: float                # 点1米制深度
    depth2_m: float                # 点2米制深度
    distance_3d: float             # 3D欧氏距离 (m)
    distance_2d_image: float       # 图像平面像素距离
    # 不确定性（如果启用）
    uncertainty_95: Optional[float] = None


def measure_two_points(
    p1_2d: Tuple[float, float],
    p2_2d: Tuple[float, float],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    metric_converter=None,
    sampling_radius: int = 3,
    percentile_low: float = 10,
    percentile_high: float = 90,
) -> Optional[TwoPointMeasurement]:
    """点击图像上两点，计算它们在三维空间中的欧氏距离

    完整流程：
      1. 在每个点击点的邻域内采样深度值（剔除异常值后取中位数）
      2. 通过 metric_converter 将深度值转为米制
      3. 针孔相机逆投影：pixel(u,v) + depth(Zc) → 3D(Xc, Yc, Zc)
      4. 计算两点间的三维欧氏距离

    Args:
        p1_2d: 第一个点的像素坐标 (u1, v1)
        p2_2d: 第二个点的像素坐标 (u2, v2)
        depth_map: 深度图 (H, W)，可以是相对值或已转换的米制值
        intrinsics: 相机内参
        metric_converter: 深度→米制转换函数，如果 depth_map 已经是米制则传 None
        sampling_radius: 邻域采样半径（像素），默认3，即7×7窗口
        percentile_low: 邻域深度值截断下界百分位
        percentile_high: 邻域深度值截断上界百分位

    Returns:
        TwoPointMeasurement 或 None（任一点深度无效时）

    Example:
        >>> result = measure_two_points(
        ...     (1200, 800), (1800, 900),
        ...     depth_map, intrinsics, metric_converter=calibrator.converter
        ... )
        >>> print(f"3D距离: {result.distance_3d:.2f}m")
    """
    H, W = depth_map.shape

    def _get_metric_depth(u: float, v: float) -> Tuple[float, float]:
        """在 (u,v) 邻域内鲁棒采样深度，返回 (raw_depth, metric_depth)"""
        ui, vi = int(round(u)), int(round(v))
        samples_raw = []
        for dy in range(-sampling_radius, sampling_radius + 1):
            for dx in range(-sampling_radius, sampling_radius + 1):
                nx, ny = ui + dx, vi + dy
                if 0 <= nx < W and 0 <= ny < H:
                    d = depth_map[ny, nx]
                    if d > 0:
                        samples_raw.append(float(d))

        if not samples_raw:
            return 0.0, 0.0

        # 百分位截断：剔除过近/过远的离群值
        lo = np.percentile(samples_raw, percentile_low)
        hi = np.percentile(samples_raw, percentile_high)
        filtered = [s for s in samples_raw if lo <= s <= hi]
        if not filtered:
            filtered = samples_raw

        raw_d = float(np.median(filtered))
        if metric_converter is not None:
            metric_d = metric_converter(raw_d)
        else:
            metric_d = raw_d
        return raw_d, metric_d

    # 采样两点深度
    raw1, metric1 = _get_metric_depth(p1_2d[0], p1_2d[1])
    raw2, metric2 = _get_metric_depth(p2_2d[0], p2_2d[1])

    if metric1 <= 0 or metric2 <= 0:
        return None

    # 逆投影到3D
    P1 = pixel_to_camera(p1_2d[0], p1_2d[1], metric1, intrinsics)
    P2 = pixel_to_camera(p2_2d[0], p2_2d[1], metric2, intrinsics)

    # 3D欧氏距离
    dist_3d = float(np.linalg.norm(P2 - P1))

    # 图像平面像素距离
    dist_2d = float(np.sqrt((p2_2d[0] - p1_2d[0]) ** 2 + (p2_2d[1] - p1_2d[1]) ** 2))

    return TwoPointMeasurement(
        point1_2d=p1_2d,
        point2_2d=p2_2d,
        point1_3d=P1,
        point2_3d=P2,
        depth1_m=metric1,
        depth2_m=metric2,
        distance_3d=dist_3d,
        distance_2d_image=dist_2d,
    )


def measure_two_points_with_uncertainty(
    p1_2d: Tuple[float, float],
    p2_2d: Tuple[float, float],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    metric_converter=None,
    calibration_rmse: float = 0.0,
    n_samples: int = 500,
    pixel_noise: float = 2.0,
    depth_noise_relative: float = 0.02,
    sampling_radius: int = 3,
) -> Dict:
    """带蒙特卡洛不确定性估计的两点测距

    对两个点击点同时施加像素噪声和深度噪声，生成 n_samples 个
    距离估计值，返回中位数和 95% 置信区间。

    Args:
        p1_2d, p2_2d: 两点像素坐标
        depth_map: 原始深度图（相对值）
        intrinsics: 相机内参
        metric_converter: 深度→米制转换
        calibration_rmse: 标定 RMSE (m)
        n_samples: 蒙特卡洛采样数
        pixel_noise: 像素噪声标准差 (px)
        depth_noise_relative: 深度值相对噪声
        sampling_radius: 邻域采样半径

    Returns:
        {'median': float, 'std': float, 'ci_95_low': float,
         'ci_95_high': float, 'best_estimate': TwoPointMeasurement}
    """
    H, W = depth_map.shape

    def _sample_raw_depth(u: float, v: float) -> float:
        ui, vi = int(round(u)), int(round(v))
        samples = []
        for dy in range(-sampling_radius, sampling_radius + 1):
            for dx in range(-sampling_radius, sampling_radius + 1):
                nx, ny = ui + dx, vi + dy
                if 0 <= nx < W and 0 <= ny < H:
                    d = depth_map[ny, nx]
                    if d > 0:
                        samples.append(float(d))
        return float(np.median(samples)) if samples else 0.0

    raw1 = _sample_raw_depth(p1_2d[0], p1_2d[1])
    raw2 = _sample_raw_depth(p2_2d[0], p2_2d[1])

    distances = []
    for _ in range(n_samples):
        # 像素位置噪声
        u1n = p1_2d[0] + np.random.normal(0, pixel_noise)
        v1n = p1_2d[1] + np.random.normal(0, pixel_noise)
        u2n = p2_2d[0] + np.random.normal(0, pixel_noise)
        v2n = p2_2d[1] + np.random.normal(0, pixel_noise)

        # 深度噪声
        d1n = raw1 * (1.0 + np.random.normal(0, depth_noise_relative))
        d2n = raw2 * (1.0 + np.random.normal(0, depth_noise_relative))
        d1n = max(d1n, 1e-6)
        d2n = max(d2n, 1e-6)

        m1 = metric_converter(d1n) + np.random.normal(0, calibration_rmse)
        m2 = metric_converter(d2n) + np.random.normal(0, calibration_rmse)
        m1 = max(m1, 0.1)
        m2 = max(m2, 0.1)

        P1n = pixel_to_camera(u1n, v1n, m1, intrinsics)
        P2n = pixel_to_camera(u2n, v2n, m2, intrinsics)
        distances.append(float(np.linalg.norm(P2n - P1n)))

    distances = np.array(distances)
    best = measure_two_points(p1_2d, p2_2d, depth_map, intrinsics,
                              metric_converter, sampling_radius)
    if best:
        best.uncertainty_95 = (np.percentile(distances, 97.5) - np.percentile(distances, 2.5)) / 2.0

    return {
        'median': float(np.median(distances)),
        'std': float(np.std(distances)),
        'ci_95_low': float(np.percentile(distances, 2.5)),
        'ci_95_high': float(np.percentile(distances, 97.5)),
        'best_estimate': best,
    }


def _point_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray
                      ) -> Tuple[float, np.ndarray]:
    """点到线段的最短距离

    Returns:
        (distance, projection_point)
    """
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / max(np.dot(AB, AB), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = A + t * AB
    return float(np.linalg.norm(P - proj)), proj
