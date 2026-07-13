"""geometric_utils — 坐标转换与几何工具

坐标系约定：
  - 像素坐标系 (u, v): u 向右, v 向下, 原点在图像左上角
  - 相机坐标系 (Xc, Yc, Zc): Xc 向右, Yc 向下, Zc 向前（深度方向）
  - 局部导线坐标系: X' 沿导线水平走向, Y' 垂直向上, Z' 垂直 X'-Y' 平面

针孔相机模型：
  u = fx * Xc / Zc + cx
  v = fy * Yc / Zc + cy

逆投影（已知深度 Zc）：
  Xc = (u - cx) * Zc / fx
  Yc = (v - cy) * Zc / fy
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int = 0, height: int = 0):
        """从 3×3 内参矩阵构造"""
        return cls(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            width=width, height=height,
        )

    @classmethod
    def from_fov(cls, hfov_deg: float, width: int, height: int):
        """从水平视场角估算内参（假设方形像素）"""
        fx = width / 2.0 / np.tan(np.radians(hfov_deg / 2.0))
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @property
    def matrix(self) -> np.ndarray:
        """返回 3×3 内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


def pixel_to_camera(u: float, v: float, depth: float,
                    intrinsics: CameraIntrinsics) -> np.ndarray:
    """像素坐标 + 深度 → 相机坐标系 3D 点

    Args:
        u, v: 像素坐标
        depth: 该点的米制深度 Zc（必须是米制，不能是相对值）
        intrinsics: 相机内参

    Returns:
        np.ndarray([Xc, Yc, Zc])  相机坐标系 3D 点
    """
    if depth <= 0:
        raise ValueError(f"深度值必须为正，当前 depth={depth}")
    Xc = (u - intrinsics.cx) * depth / intrinsics.fx
    Yc = (v - intrinsics.cy) * depth / intrinsics.fy
    return np.array([Xc, Yc, depth], dtype=np.float64)


def camera_to_pixel(P: np.ndarray,
                    intrinsics: CameraIntrinsics) -> Tuple[float, float]:
    """相机坐标系 3D 点 → 像素坐标

    Args:
        P: [Xc, Yc, Zc]
        intrinsics: 相机内参

    Returns:
        (u, v) 像素坐标
    """
    Xc, Yc, Zc = P
    if Zc <= 0:
        raise ValueError(f"Zc 必须为正，当前 Zc={Zc}")
    u = intrinsics.fx * Xc / Zc + intrinsics.cx
    v = intrinsics.fy * Yc / Zc + intrinsics.cy
    return u, v


def project_pixels_to_3d(pixel_points: List[Tuple[float, float]],
                         depth_values: np.ndarray,
                         depth_map: np.ndarray,
                         intrinsics: CameraIntrinsics,
                         metric_converter=None,
                         sampling_radius: int = 1) -> List[np.ndarray]:
    """将一组像素点反投影为 3D 点（带邻域采样）

    对每个像素点，在其邻域内采样深度值，取中位数以提高鲁棒性。

    Args:
        pixel_points: [(u1, v1), (u2, v2), ...]
        depth_values: 预采样的深度值（与 pixel_points 一一对应），如果为 None 则从 depth_map 采样
        depth_map: 完整深度图 (H × W)，相对值或米制均可，会经 metric_converter 转换
        intrinsics: 相机内参
        metric_converter: 将深度值转为米制的可调用对象
        sampling_radius: 邻域半径（像素）

    Returns:
        [np.array([X1,Y1,Z1]), ...]  相机坐标系 3D 点列表
    """
    H, W = depth_map.shape
    points_3d = []

    for i, (u, v) in enumerate(pixel_points):
        ui, vi = int(round(u)), int(round(v))
        # 邻域采样
        samples = []
        for dy in range(-sampling_radius, sampling_radius + 1):
            for dx in range(-sampling_radius, sampling_radius + 1):
                nx, ny = ui + dx, vi + dy
                if 0 <= nx < W and 0 <= ny < H:
                    d = depth_map[ny, nx]
                    if d > 0:
                        if metric_converter is not None:
                            d = metric_converter(d)
                        samples.append(d)

        if not samples:
            # 退化为单点采样
            if depth_values is not None and i < len(depth_values):
                d = depth_values[i]
                if metric_converter is not None:
                    d = metric_converter(d)
            elif 0 <= ui < W and 0 <= vi < H:
                d = depth_map[vi, ui]
                if metric_converter is not None:
                    d = metric_converter(d)
            else:
                continue
            if d <= 0:
                continue
        else:
            d = float(np.median(samples))

        points_3d.append(pixel_to_camera(u, v, d, intrinsics))

    return points_3d


def build_local_wire_frame(points_3d: List[np.ndarray]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """从导线的 3D 点构建局部坐标系

    步骤：
    1. 将 3D 点投影到 Xc-Zc 平面（水平面），RANSAC 拟合直线得到导线走向
    2. X' 轴 = 导线走向（水平面上）
    3. Y' 轴 = 相机 Yc 轴的投影（近似垂直方向，指向上方）
    4. Z' 轴 = X' × Y'（右手系）

    Args:
        points_3d: 相机坐标系下的导线 3D 点列表

    Returns:
        origin: 局部坐标系原点（导线在 XZ 平面上的起点投影）
        x_axis: X' 方向单位向量
        y_axis: Y' 方向单位向量
        z_axis: Z' 方向单位向量
        wire_length_horizontal: 导线水平投影长度
    """
    if len(points_3d) < 2:
        raise ValueError("至少需要 2 个点来构建局部坐标系")

    pts = np.array(points_3d)
    # 投影到 XZ 平面
    pts_xz = pts[:, [0, 2]]

    # RANSAC 拟合直线方向
    best_direction = _ransac_line_direction(pts_xz)

    # X' 轴：导线在水平面上的走向
    x_axis = np.array([best_direction[0], 0.0, best_direction[1]], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)

    # Y' 轴：取相机 Yc 方向，但确保与 X' 正交
    y_cam = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # 相机 Yc 向下，物理高度向上为 -Yc
    y_axis = y_cam - np.dot(y_cam, x_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = np.array([0.0, -1.0, 0.0])
    y_axis /= np.linalg.norm(y_axis)

    # Z' 轴 = X' × Y'
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # 原点：第一个点在 XZ 平面上的投影沿 X' 轴的最近点
    origin_3d = np.array([pts[0, 0], 0.0, pts[0, 2]], dtype=np.float64)

    # 计算水平投影长度
    projections = np.dot(pts_xz - pts_xz[0], best_direction)
    wire_length_horizontal = float(projections[-1] - projections[0])
    if wire_length_horizontal < 0:
        # 确保方向一致性
        x_axis = -x_axis
        z_axis = np.cross(x_axis, y_axis)
        wire_length_horizontal = -wire_length_horizontal

    return origin_3d, x_axis, y_axis, z_axis, wire_length_horizontal


def _ransac_line_direction(pts_2d: np.ndarray, n_iter: int = 200,
                           inlier_thresh: float = 0.3) -> np.ndarray:
    """RANSAC 拟合 2D 点集的直线方向向量

    Args:
        pts_2d: (N, 2) 点集
        n_iter: 迭代次数
        inlier_thresh: 内点距离阈值

    Returns:
        归一化方向向量 (2,)
    """
    best_inliers = 0
    best_dir = None
    N = len(pts_2d)

    if N < 2:
        return np.array([1.0, 0.0])

    for _ in range(n_iter):
        i, j = np.random.choice(N, size=2, replace=False)
        direction = pts_2d[j] - pts_2d[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction /= norm
        # 计算各点到该直线的距离
        vecs = pts_2d - pts_2d[i]
        # 手动 2D 叉积（兼容 numpy>=2.0，该版本 np.cross 要求 1D 向量）
        dists = np.abs(vecs[:, 0] * direction[1] - vecs[:, 1] * direction[0])
        n_inliers = np.sum(dists < inlier_thresh)
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_dir = direction

    if best_dir is None:
        # fallback: PCA
        centered = pts_2d - np.mean(pts_2d, axis=0)
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        best_dir = eigvecs[:, -1]

    return best_dir / np.linalg.norm(best_dir)


def transform_to_local(points_3d: List[np.ndarray],
                       origin: np.ndarray,
                       x_axis: np.ndarray,
                       y_axis: np.ndarray,
                       z_axis: np.ndarray) -> np.ndarray:
    """将相机坐标系 3D 点转换到局部导线坐标系

    Args:
        points_3d: 相机坐标系 3D 点列表
        origin, x_axis, y_axis, z_axis: 局部坐标系定义

    Returns:
        (N, 3) 数组，列为 [X', Y', Z']
    """
    R = np.column_stack([x_axis, y_axis, z_axis])  # 3×3 旋转矩阵
    pts = np.array(points_3d)
    local = (pts - origin) @ R
    return local


def sample_depth_in_bbox(depth_map: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         metric_converter=None,
                         percentile_low: float = 10,
                         percentile_high: float = 90) -> float:
    """在边界框内鲁棒采样深度值

    剔除异常值后取中位数作为该物体的代表深度。

    Args:
        depth_map: 深度图 (H, W)
        bbox: (x_min, y_min, x_max, y_max)
        metric_converter: 深度→米制转换函数
        percentile_low, percentile_high: 截断百分位

    Returns:
        该物体的代表深度（米制，如果提供了 converter）
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max = min(depth_map.shape[1], x_max)
    y_max = min(depth_map.shape[0], y_max)

    region = depth_map[y_min:y_max, x_min:x_max]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0

    lo = np.percentile(valid, percentile_low)
    hi = np.percentile(valid, percentile_high)
    filtered = valid[(valid >= lo) & (valid <= hi)]
    if len(filtered) == 0:
        filtered = valid

    depth = float(np.median(filtered))
    if metric_converter is not None:
        depth = metric_converter(depth)
    return depth
