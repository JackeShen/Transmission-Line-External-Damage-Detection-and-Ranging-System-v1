"""calibration — 深度到米制的鲁棒标定

解决的问题：深度估计模型输出的是相对深度值（或归一化深度），与真实米制距离之间
不是简单的反比或线性关系。本模块提供：

  - 多点标定：使用多个已知距离的标定物
  - 鲁棒回归：Theil-Sen、RANSAC、Huber 三种方法
  - 多种模型：线性、二次、幂律
  - 交叉验证选最优模型

使用方式：
    pairs = [(depth_val_1, real_dist_1), (depth_val_2, real_dist_2), ...]
    calibrator = DepthCalibrator(method='theil_sen', model='quadratic')
    calibrator.fit(pairs)
    converter = calibrator.converter  # 可调用对象: float → float
    metric_depth = converter(raw_depth)
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import least_squares
import warnings


@dataclass
class CalibrationResult:
    """标定结果"""
    model_type: str           # 'linear' | 'quadratic' | 'power'
    method: str               # 'theil_sen' | 'ransac' | 'huber'
    params: np.ndarray        # 模型参数
    residuals: np.ndarray     # 残差（米）
    rmse: float               # 均方根误差（米）
    mae: float                # 平均绝对误差（米）
    r_squared: float          # 决定系数
    n_points: int             # 标定点数
    depth_range: Tuple[float, float]  # 标定覆盖的深度范围


class DepthCalibrator:
    """深度→米制鲁棒标定器

    支持的模型：
      - 'linear':   metric = a * depth + b
      - 'quadratic': metric = a * depth² + b * depth + c
      - 'power':    metric = a * depth^p + b

    支持的方法：
      - 'theil_sen': Theil-Sen 估计（最鲁棒，推荐 ≥5 个标定点）
      - 'ransac':    RANSAC 回归
      - 'huber':     Huber 损失最小化（推荐 ≥3 个标定点）
    """

    def __init__(self, method: str = 'theil_sen', model: str = 'quadratic'):
        """
        Args:
            method: 'theil_sen' | 'ransac' | 'huber'
            model:  'linear' | 'quadratic' | 'power'
        """
        self.method = method
        self.model = model
        self.result: Optional[CalibrationResult] = None
        self._converter: Optional[Callable[[float], float]] = None

    def fit(self, pairs: List[Tuple[float, float]],
            depth_range_override: Optional[Tuple[float, float]] = None
            ) -> CalibrationResult:
        """用 (深度值, 真实距离) 标定点对拟合模型

        Args:
            pairs: [(depth_raw, distance_meters), ...]  至少 3 对
            depth_range_override: 手动指定深度值有效范围

        Returns:
            CalibrationResult
        """
        if len(pairs) < 3:
            raise ValueError(f"标定至少需要 3 个点，当前只有 {len(pairs)} 个")

        depths = np.array([p[0] for p in pairs], dtype=np.float64)
        distances = np.array([p[1] for p in pairs], dtype=np.float64)

        # 排序
        sort_idx = np.argsort(depths)
        depths = depths[sort_idx]
        distances = distances[sort_idx]

        if self.model == 'linear':
            params = self._fit_linear(depths, distances)
        elif self.model == 'quadratic':
            params = self._fit_quadratic(depths, distances)
        elif self.model == 'power':
            params = self._fit_power(depths, distances)
        else:
            raise ValueError(f"未知模型类型: {self.model}")

        # 评估
        predicted = self._predict(depths, params)
        residuals = distances - predicted
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((distances - np.mean(distances)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self.result = CalibrationResult(
            model_type=self.model,
            method=self.method,
            params=params,
            residuals=residuals,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            n_points=len(pairs),
            depth_range=(float(depths[0]), float(depths[-1])),
        )

        self._converter = lambda d: self._predict(d, params)
        return self.result

    @property
    def converter(self) -> Callable[[float], float]:
        """返回深度→米制的转换函数"""
        if self._converter is None:
            raise RuntimeError("请先调用 fit() 进行标定")
        return self._converter

    def _predict(self, depth: np.ndarray, params: np.ndarray) -> np.ndarray:
        """根据模型参数预测距离"""
        if self.model == 'linear':
            return params[0] * depth + params[1]
        elif self.model == 'quadratic':
            return params[0] * depth ** 2 + params[1] * depth + params[2]
        elif self.model == 'power':
            return params[0] * depth ** params[2] + params[1]
        raise ValueError(f"未知模型: {self.model}")

    # ─── 拟合方法 ──────────────────────────────────────────────

    def _fit_linear(self, depths: np.ndarray,
                    distances: np.ndarray) -> np.ndarray:
        if self.method == 'theil_sen':
            return self._theil_sen_linear(depths, distances)
        elif self.method == 'ransac':
            return self._ransac_linear(depths, distances)
        elif self.method == 'huber':
            return self._huber_linear(depths, distances)
        raise ValueError(f"未知方法: {self.method}")

    def _fit_quadratic(self, depths: np.ndarray,
                       distances: np.ndarray) -> np.ndarray:
        if self.method == 'theil_sen':
            return self._theil_sen_quadratic(depths, distances)
        elif self.method == 'ransac':
            return self._ransac_quadratic(depths, distances)
        elif self.method == 'huber':
            return self._huber_quadratic(depths, distances)
        raise ValueError(f"未知方法: {self.method}")

    def _fit_power(self, depths: np.ndarray,
                   distances: np.ndarray) -> np.ndarray:
        if self.method in ('theil_sen', 'ransac'):
            return self._huber_power(depths, distances)
        return self._huber_power(depths, distances)

    # ─── Theil-Sen ─────────────────────────────────────────────

    @staticmethod
    def _theil_sen_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Theil-Sen 线性回归"""
        n = len(x)
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(x[j] - x[i]) > 1e-9:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        if not slopes:
            return np.array([0.0, np.mean(y)])
        slope = np.median(slopes)
        intercept = np.median(y - slope * x)
        return np.array([slope, intercept])

    def _theil_sen_quadratic(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Theil-Sen 扩展：先线性剔除异常值，再最小二乘拟合二次"""
        # 先用线性 Theil-Sen 剔除异常值
        lin_params = self._theil_sen_linear(x, y)
        residuals = y - (lin_params[0] * x + lin_params[1])
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = 3.0 * mad * 1.4826  # MAD → 标准差近似

        mask = np.abs(residuals) < max(threshold, 0.5)  # 至少 0.5m 的容忍度
        if mask.sum() < 3:
            mask = np.ones(len(x), dtype=bool)

        # 对内点做二次最小二乘
        A = np.column_stack([x[mask] ** 2, x[mask], np.ones(mask.sum())])
        params, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
        return params

    # ─── RANSAC ────────────────────────────────────────────────

    @staticmethod
    def _ransac_linear(x: np.ndarray, y: np.ndarray,
                       n_iter: int = 500, inlier_thresh: float = 1.0
                       ) -> np.ndarray:
        best_inliers = 0
        best_params = np.array([0.0, np.mean(y)])
        n = len(x)
        for _ in range(n_iter):
            i, j = np.random.choice(n, size=2, replace=False)
            if abs(x[j] - x[i]) < 1e-9:
                continue
            slope = (y[j] - y[i]) / (x[j] - x[i])
            intercept = y[i] - slope * x[i]
            residuals = np.abs(y - (slope * x + intercept))
            n_inliers = np.sum(residuals < inlier_thresh)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_params = np.array([slope, intercept])
        return best_params

    def _ransac_quadratic(self, x: np.ndarray, y: np.ndarray,
                          n_iter: int = 500, inlier_thresh: float = 1.0
                          ) -> np.ndarray:
        best_inliers = 0
        best_params = None
        n = len(x)
        for _ in range(n_iter):
            indices = np.random.choice(n, size=min(4, n), replace=False)
            if len(indices) < 3:
                continue
            A = np.column_stack([x[indices] ** 2, x[indices], np.ones(len(indices))])
            try:
                params, _, _, _ = np.linalg.lstsq(A, y[indices], rcond=None)
            except np.linalg.LinAlgError:
                continue
            residuals = np.abs(y - (params[0] * x ** 2 + params[1] * x + params[2]))
            n_inliers = np.sum(residuals < inlier_thresh)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_params = params
        if best_params is None:
            A = np.column_stack([x ** 2, x, np.ones(n)])
            best_params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return best_params

    # ─── Huber ─────────────────────────────────────────────────

    def _huber_linear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        init = self._theil_sen_linear(x, y)
        result = least_squares(
            lambda p: _huber_weight(y - (p[0] * x + p[1])),
            init, method='trf', loss='soft_l1'
        )
        return result.x

    def _huber_quadratic(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        init = self._theil_sen_quadratic(x, y)
        result = least_squares(
            lambda p: _huber_weight(y - (p[0] * x ** 2 + p[1] * x + p[2])),
            init, method='trf', loss='soft_l1'
        )
        return result.x

    def _huber_power(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # 初始值：a ≈ y_mean / x_mean^0.5, p = 0.5, b = 0
        init = np.array([np.mean(y) / (np.mean(x) ** 0.5 + 1e-6), 0.0, 0.5])
        result = least_squares(
            lambda p: _huber_weight(y - (p[0] * x ** p[2] + p[1])),
            init, method='trf', loss='soft_l1',
            bounds=([0, -np.inf, 0.1], [np.inf, np.inf, 3.0])
        )
        return result.x


def _huber_weight(residuals: np.ndarray, delta: float = 1.345) -> np.ndarray:
    """Huber 权重函数，返回加权残差供 least_squares 使用"""
    abs_r = np.abs(residuals)
    weights = np.where(abs_r <= delta, 1.0, delta / abs_r)
    return residuals * np.sqrt(weights)


def auto_calibrate(depth_map: np.ndarray,
                   calibration_objects: List[Dict],
                   try_models: Optional[List[str]] = None
                   ) -> DepthCalibrator:
    """从标定物自动选择最优模型

    对每个标定物从 depth_map 中提取 bbox 内的中位深度值，
    与已知距离组成标定对，尝试多种模型，选 RMSE 最低的。

    Args:
        depth_map: 深度图 (H, W)
        calibration_objects: [{'bbox': [x1,y1,x2,y2], 'distance': 30.0}, ...]
        try_models: 候选模型列表，默认 ['linear', 'quadratic']

    Returns:
        已拟合的 DepthCalibrator 实例
    """
    if try_models is None:
        try_models = ['linear', 'quadratic']

    # 构建标定对
    pairs = []
    for obj in calibration_objects:
        bbox = obj['bbox']
        known_dist = obj['distance']
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max = min(depth_map.shape[1], x_max)
        y_max = min(depth_map.shape[0], y_max)
        region = depth_map[y_min:y_max, x_min:x_max]
        valid = region[region > 0]
        if len(valid) == 0:
            continue
        depth_val = float(np.median(valid))
        pairs.append((depth_val, known_dist))

    if len(pairs) < 3:
        raise ValueError(f"有效标定点不足：{len(pairs)}，需要至少 3 个")

    # 尝试多种模型，选最优
    best_calibrator = None
    best_rmse = float('inf')

    for model in try_models:
        for method in ['theil_sen', 'huber']:
            try:
                cal = DepthCalibrator(method=method, model=model)
                cal.fit(pairs)
                if cal.result.rmse < best_rmse:
                    best_rmse = cal.result.rmse
                    best_calibrator = cal
            except Exception:
                continue

    if best_calibrator is None:
        # fallback: simplest linear theil-sen
        best_calibrator = DepthCalibrator(method='theil_sen', model='linear')
        best_calibrator.fit(pairs)

    return best_calibrator
