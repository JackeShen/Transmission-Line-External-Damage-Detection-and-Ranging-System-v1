"""catenary — 物理正确的悬链线拟合

与旧版 spatial_analyzer.py 中 _catenary_fit() 的根本区别：

  旧版：在像素/深度坐标系中拟合 z = a·cosh((x-c)/a) + d
        → 深度 Z 作为因变量，无物理意义

  新版：在局部导线坐标系中拟合 Y' = a·cosh((X'-x₀)/a) + b
        → 高度 Y' 沿导线水平走向 X' 的悬链线
        → 支持杆塔挂点硬约束
        → 输出密集采样点供 KD-Tree 使用

坐标系：
  局部导线坐标系由 geometric_utils.build_local_wire_frame() 定义：
    X' = 沿导线水平走向（在相机 Xc-Zc 平面内）
    Y' = 近似垂直方向（≈ -Yc，即物理高度方向）
    Z' = X' × Y'（垂直导线平面）
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from scipy.optimize import minimize, curve_fit
from scipy.spatial import KDTree

from .geometric_utils import build_local_wire_frame, transform_to_local


@dataclass
class CatenaryModel:
    """悬链线拟合结果"""
    wire_id: str
    # 悬链线参数：Y' = a * cosh((X' - x0) / a) + b
    a: float         # 水平张力 / 单位长度重量 (m)，典型值 500~3000
    x0: float        # 最低点在 X' 轴上的位置 (m)
    b: float         # 最低点高度偏移 (m)
    # 几何信息
    span_length: float        # 档距（水平投影长度，m）
    sag: float                # 最大弧垂 (m)
    # 局部坐标系变换
    origin: np.ndarray        # 局部坐标系原点（相机坐标）
    x_axis: np.ndarray        # X' 方向单位向量（相机坐标）
    y_axis: np.ndarray        # Y' 方向单位向量（相机坐标）
    z_axis: np.ndarray        # Z' 方向单位向量（相机坐标）
    # 拟合质量
    fit_rms: float            # 拟合 RMS 误差 (m)
    n_points_used: int        # 参与拟合的点数
    # 密集采样
    samples_3d: np.ndarray    # (M, 3) 相机坐标系下的密集采样点
    sample_spacing: float = 0.1  # 采样间距 (m)
    kd_tree: Optional[KDTree] = None

    def build_kd_tree(self):
        """构建 KD-Tree"""
        self.kd_tree = KDTree(self.samples_3d)


def fit_catenary(points_3d: List[np.ndarray],
                 wire_id: str = "wire",
                 tower_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 sample_spacing: float = 0.1,
                 a_bounds: Tuple[float, float] = (100.0, 10000.0),
                 ) -> Optional[CatenaryModel]:
    """从导线 3D 点拟合悬链线

    Args:
        points_3d: 相机坐标系下的导线 3D 点列表
        wire_id: 导线标识
        tower_endpoints: 可选的两端杆塔挂点（相机坐标系），(P_left, P_right)
                         如果提供，会作为硬约束
        sample_spacing: 输出采样点间距 (m)
        a_bounds: 参数 a 的有效范围 (m)

    Returns:
        CatenaryModel 或 None（拟合失败时）
    """
    if len(points_3d) < 5:
        print(f"  [catenary] {wire_id}: 点数不足 ({len(points_3d)} < 5)，跳过")
        return None

    pts = np.array(points_3d)

    # 步骤 1: 构建局部坐标系
    try:
        origin, x_axis, y_axis, z_axis, span_len = build_local_wire_frame(points_3d)
    except ValueError as e:
        print(f"  [catenary] {wire_id}: 构建局部坐标系失败 — {e}")
        return None

    # 步骤 2: 变换到局部坐标
    local_pts = transform_to_local(points_3d, origin, x_axis, y_axis, z_axis)
    X_prime = local_pts[:, 0]   # 沿导线方向
    Y_prime = local_pts[:, 1]   # 高度方向

    # 步骤 3: 初始参数估计
    a_init, x0_init, b_init = _estimate_catenary_init(
        X_prime, Y_prime, span_len, tower_endpoints, origin, x_axis, y_axis, z_axis
    )

    # 步骤 4: 拟合
    if tower_endpoints is not None:
        params = _fit_catenary_constrained(
            X_prime, Y_prime, a_init, x0_init, b_init,
            tower_endpoints, origin, x_axis, y_axis, z_axis, a_bounds
        )
    else:
        params = _fit_catenary_unconstrained(
            X_prime, Y_prime, a_init, x0_init, b_init, a_bounds
        )

    a_fit, x0_fit, b_fit = params

    # 步骤 5: 评估拟合质量
    Y_pred = a_fit * np.cosh((X_prime - x0_fit) / a_fit) + b_fit
    residuals = Y_prime - Y_pred
    rms = np.sqrt(np.mean(residuals ** 2))

    # 如果拟合太差，可能是数据质量问题
    if rms > 1.0:
        print(f"  [catenary] {wire_id}: 拟合 RMS={rms:.2f}m > 1.0m，结果可能不可靠")

    # 步骤 6: 计算弧垂
    if tower_endpoints is not None:
        # 从塔端点计算跨度
        P_left, P_right = tower_endpoints
        left_local = transform_to_local([P_left], origin, x_axis, y_axis, z_axis)[0]
        right_local = transform_to_local([P_right], origin, x_axis, y_axis, z_axis)[0]
        x_min_fit = left_local[0]
        x_max_fit = right_local[0]
    else:
        x_min_fit = X_prime.min()
        x_max_fit = X_prime.max()

    span = x_max_fit - x_min_fit
    # 弧垂：端点连线的中点高度 - 悬链线在中点的高度
    Y_left = a_fit * np.cosh((x_min_fit - x0_fit) / a_fit) + b_fit
    Y_right = a_fit * np.cosh((x_max_fit - x0_fit) / a_fit) + b_fit
    Y_mid_straight = (Y_left + Y_right) / 2.0
    x_mid = (x_min_fit + x_max_fit) / 2.0
    Y_mid_catenary = a_fit * np.cosh((x_mid - x0_fit) / a_fit) + b_fit
    sag = Y_mid_straight - Y_mid_catenary

    # 步骤 7: 密集采样（在局部坐标系）
    n_samples = max(int(span / sample_spacing) + 1, 10)
    X_sample = np.linspace(x_min_fit, x_max_fit, n_samples)
    Y_sample = a_fit * np.cosh((X_sample - x0_fit) / a_fit) + b_fit
    Z_sample = np.zeros_like(X_sample)

    # 变换回相机坐标系
    local_samples = np.column_stack([X_sample, Y_sample, Z_sample])
    R_inv = np.column_stack([x_axis, y_axis, z_axis])
    samples_3d = local_samples @ R_inv.T + origin

    model = CatenaryModel(
        wire_id=wire_id,
        a=a_fit, x0=x0_fit, b=b_fit,
        span_length=span,
        sag=sag,
        origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        fit_rms=rms,
        n_points_used=len(points_3d),
        samples_3d=samples_3d,
        sample_spacing=sample_spacing,
    )
    model.build_kd_tree()

    print(f"  [catenary] {wire_id}: a={a_fit:.1f}m, span={span:.1f}m, "
          f"sag={sag:.2f}m, RMS={rms:.3f}m")
    return model


# ═══════════════════════════════════════════════════════════════
#  内部辅助函数
# ═══════════════════════════════════════════════════════════════

def _estimate_catenary_init(X: np.ndarray, Y: np.ndarray,
                            span: float,
                            tower_endpoints, origin, x_axis, y_axis, z_axis
                            ) -> Tuple[float, float, float]:
    """估算悬链线初始参数"""
    x_min, x_max = X.min(), X.max()

    if tower_endpoints is not None:
        P_left, P_right = tower_endpoints
        left_local = transform_to_local([P_left], origin, x_axis, y_axis, z_axis)[0]
        right_local = transform_to_local([P_right], origin, x_axis, y_axis, z_axis)[0]
        y_left = left_local[1]
        y_right = right_local[1]
    else:
        # 用端点附近的 Y 均值估计挂点高度
        margin = max(1, int(len(X) * 0.05))
        y_left = np.mean(Y[:margin])
        y_right = np.mean(Y[-margin:])

    y_mid = np.min(Y)
    y_end_mean = (y_left + y_right) / 2.0
    f_max = max(y_end_mean - y_mid, 0.5)

    # a ≈ L² / (8·f)  (抛物线近似)
    a_init = np.clip(span ** 2 / (8.0 * f_max), 200.0, 5000.0)
    x0_init = (x_min + x_max) / 2.0
    b_init = y_mid - a_init

    return a_init, x0_init, b_init


def _catenary_func(X: np.ndarray, a: float, x0: float, b: float) -> np.ndarray:
    """悬链线函数"""
    return a * np.cosh((X - x0) / a) + b


def _fit_catenary_unconstrained(X: np.ndarray, Y: np.ndarray,
                                a_init: float, x0_init: float, b_init: float,
                                a_bounds: Tuple[float, float]
                                ) -> np.ndarray:
    """无约束悬链线拟合（用 curve_fit）"""
    try:
        popt, _ = curve_fit(
            _catenary_func, X, Y,
            p0=[a_init, x0_init, b_init],
            bounds=([a_bounds[0], X.min(), -np.inf],
                     [a_bounds[1], X.max(), np.inf]),
            max_nfev=2000,
        )
        return popt
    except Exception:
        # 回退到初始估计
        return np.array([a_init, x0_init, b_init])


def _fit_catenary_constrained(X: np.ndarray, Y: np.ndarray,
                              a_init: float, x0_init: float, b_init: float,
                              tower_endpoints: Tuple[np.ndarray, np.ndarray],
                              origin, x_axis, y_axis, z_axis,
                              a_bounds: Tuple[float, float]
                              ) -> np.ndarray:
    """带杆塔硬约束的悬链线拟合

    硬约束：悬链线必须通过两个挂点
    令挂点 (x_L, y_L) 和 (x_R, y_R)，有：
      y_L = a·cosh((x_L - x0)/a) + b
      y_R = a·cosh((x_R - x0)/a) + b
    → b = y_L - a·cosh((x_L - x0)/a)
    然后拟合 x0 和 a（b 由约束确定）
    """
    P_left, P_right = tower_endpoints
    left_local = transform_to_local([P_left], origin, x_axis, y_axis, z_axis)[0]
    right_local = transform_to_local([P_right], origin, x_axis, y_axis, z_axis)[0]
    xL, yL = left_local[0], left_local[1]
    xR, yR = right_local[0], right_local[1]

    def objective(params):
        a, x0 = params
        b = yL - a * np.cosh((xL - x0) / a)
        Y_pred = a * np.cosh((X - x0) / a) + b
        return np.sum((Y - Y_pred) ** 2)

    result = minimize(
        objective,
        x0=[a_init, x0_init],
        bounds=[(a_bounds[0], a_bounds[1]), (xL, xR)],
        method='L-BFGS-B',
    )

    a_opt, x0_opt = result.x
    b_opt = yL - a_opt * np.cosh((xL - x0_opt) / a_opt)
    return np.array([a_opt, x0_opt, b_opt])


def fit_all_wires(wire_groups: Dict[str, List[np.ndarray]],
                  tower_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                  sample_spacing: float = 0.1
                  ) -> List[CatenaryModel]:
    """为所有导线分组拟合悬链线

    Args:
        wire_groups: {wire_id: [3d_point_1, 3d_point_2, ...]}
        tower_data: {wire_id: (tower_left_3d, tower_right_3d)} 或 None
        sample_spacing: 采样间距

    Returns:
        拟合成功的 CatenaryModel 列表
    """
    models = []
    for wire_id, points in wire_groups.items():
        endpoints = tower_data.get(wire_id) if tower_data else None
        model = fit_catenary(
            points, wire_id=wire_id,
            tower_endpoints=endpoints,
            sample_spacing=sample_spacing,
        )
        if model is not None:
            models.append(model)
    return models
