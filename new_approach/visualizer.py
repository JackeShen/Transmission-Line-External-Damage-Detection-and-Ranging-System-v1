"""visualizer — 2D 与 3D 可视化

输出：
  - 2D 图像叠加：标注框、距离连线、安全等级标签
  - 3D 点云：导线悬链线、危险物位置、最近距离标注
  - 交互式点击测距：鼠标点击图像上两点，实时计算3D距离
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple, Callable
import cv2

from .distance import (
    DistanceResult,
    measure_two_points,
    measure_two_points_with_uncertainty,
    TwoPointMeasurement,
)
from .catenary import CatenaryModel
from .geometric_utils import CameraIntrinsics

# 安全等级配色
LEVEL_COLORS = {
    "SAFE":    "#2ecc71",
    "WARNING": "#f39c12",
    "DANGER":  "#e74c3c",
    "CRITICAL":"#8e44ad",
}


def draw_2d_overlay(image: np.ndarray,
                    results: List[DistanceResult],
                    wire_points_2d: Optional[List[Tuple[List, str]]] = None,
                    output_path: Optional[str] = None,
                    show_uncertainty: bool = True,
                    ) -> np.ndarray:
    """在原始图像上叠加测距结果

    Args:
        image: RGB 图像 (H, W, 3)
        results: 测距结果列表
        wire_points_2d: [(points_list, wire_id), ...]  导线 2D 点
        output_path: 保存路径
        show_uncertainty: 是否显示不确定性区间

    Returns:
        绘制后的图像数组
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)

    # 绘制导线
    wire_colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']
    if wire_points_2d:
        for i, (pts, wid) in enumerate(wire_points_2d):
            pts_arr = np.array(pts)
            color = wire_colors[i % len(wire_colors)]
            ax.plot(pts_arr[:, 0], pts_arr[:, 1], '-', color=color,
                    linewidth=2, alpha=0.8, label=f'Wire {wid}')

    # 绘制危险物和测距结果
    for r in results:
        x1, y1, x2, y2 = r.bbox
        w, h = x2 - x1, y2 - y1
        color = LEVEL_COLORS.get(r.safety_level, '#95a5a6')
        u, v = r.center_2d

        # 边界框
        rect = Rectangle((x1, y1), w, h, fill=False,
                          edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)

        # ---- 连线: 危险物中心 → 最近导线点 ----
        best_pt = None
        if wire_points_2d:
            best_dist_2d = float('inf')
            for pts, wid in wire_points_2d:
                for px, py in pts:
                    d2 = (u - px)**2 + (v - py)**2
                    if d2 < best_dist_2d:
                        best_dist_2d = d2
                        best_pt = (px, py)
            if best_pt:
                ax.plot([u, best_pt[0]], [v, best_pt[1]], '--',
                        color=color, linewidth=2, alpha=0.7)
                ax.plot(best_pt[0], best_pt[1], 'o', color=color,
                        markersize=6, alpha=0.8)

        # 距离标注 (放在连线中点)
        mx = (u + best_pt[0]) / 2 if best_pt else u
        my = (v + best_pt[1]) / 2 - 6 if best_pt else y1 - 8
        dist_text = f"{r.distance:.1f}m"
        if r.uncertainty_95 and show_uncertainty:
            dist_text += f" {r.uncertainty_95:.1f}"

        ax.text(mx, my, dist_text, color='white',
                backgroundcolor=color, fontsize=9, fontweight='bold',
                ha='center', va='bottom')

        # 安全等级标签
        ax.text(x2 + 4, y1, r.safety_level, color='white',
                backgroundcolor=color, fontsize=7, fontweight='bold',
                ha='left', va='top')

        # 物体类别 ID
        if r.class_id is not None:
            ax.text(x1 + 2, y1 + 2, f"#{r.class_id}", color='white',
                    backgroundcolor='black', fontsize=6, alpha=0.7,
                    ha='left', va='top')

    # 图例
    handles = []
    for level, color in LEVEL_COLORS.items():
        handles.append(Patch(color=color, label=level))
    if wire_points_2d:
        handles.append(Line2D([0], [0], color='#3498db', linewidth=2,
                               label='Power Lines'))
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8,
                  framealpha=0.9)

    ax.set_title('Distance Measurement Results', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  2D 可视化已保存: {output_path}")

    # 返回图像用于后续组合
    fig.canvas.draw()
    overlay = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return overlay


def draw_3d_point_cloud(results: List[DistanceResult],
                        wire_models: List[CatenaryModel],
                        output_path: Optional[str] = None,
                        elev: float = 25, azim: float = -60,
                        ) -> None:
    """3D 点云可视化

    绘制导线悬链线（密集采样）、危险物位置、最近距离连线。

    Args:
        results: 测距结果
        wire_models: 悬链线模型
        output_path: 保存路径
        elev, azim: 视角参数
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 导线
    wire_colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']
    for i, wm in enumerate(wire_models):
        color = wire_colors[i % len(wire_colors)]
        pts = wm.samples_3d
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, linewidth=2, alpha=0.85,
                label=f'Wire {wm.wire_id}'
                      f'\n  a={wm.a:.0f}m, sag={wm.sag:.2f}m')

    # 危险物
    for r in results:
        color = LEVEL_COLORS.get(r.safety_level, '#95a5a6')
        P = r.center_3d
        ax.scatter(*P, c=color, s=80, marker='s', edgecolors='black',
                   linewidth=0.5, zorder=10)
        ax.text(P[0], P[1], P[2], f" #{r.object_id}", fontsize=7, color=color)

    # 标注坐标系
    all_pts = []
    for wm in wire_models:
        all_pts.append(wm.samples_3d)
    for r in results:
        all_pts.append(r.center_3d.reshape(1, 3))

    if all_pts:
        all_pts = np.vstack(all_pts)
        x_mid = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        y_mid = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
        z_mid = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
        span = max(np.ptp(all_pts, axis=0)) * 0.6

    ax.set_xlabel('X (m) →')
    ax.set_ylabel('Y (m) ↓')
    ax.set_zlabel('Z (m) →')
    ax.set_title('3D Scene — Wire Catenaries & Hazard Objects',
                 fontsize=13, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  3D 点云已保存: {output_path}")

    plt.close(fig)


def draw_combined(image_path: str,
                  results: List[DistanceResult],
                  wire_models: List[CatenaryModel],
                  wire_points_2d: Optional[List[Tuple[List, str]]] = None,
                  output_path: Optional[str] = None,
                  ) -> str:
    """生成 2D + 3D 组合图

    Returns:
        输出路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(18, 8))

    # 左：2D 叠加
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    if wire_points_2d:
        wire_colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']
        for i, (pts, wid) in enumerate(wire_points_2d):
            pts_arr = np.array(pts)
            ax1.plot(pts_arr[:, 0], pts_arr[:, 1], '-',
                     color=wire_colors[i % len(wire_colors)], linewidth=2, alpha=0.8)
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = LEVEL_COLORS.get(r.safety_level, '#95a5a6')
        ax1.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=color, linewidth=2))
        dist_text = f"{r.distance:.1f}m"
        if r.uncertainty_95:
            dist_text += f" ±{r.uncertainty_95:.1f}"
        ax1.text(r.center_2d[0], y1 - 6, dist_text,
                 color='white', backgroundcolor=color, fontsize=8,
                 fontweight='bold', ha='center', va='bottom')
    ax1.set_title('2D — Detection & Distance', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 右：3D 点云
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    wire_colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']
    for i, wm in enumerate(wire_models):
        pts = wm.samples_3d
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                 color=wire_colors[i % len(wire_colors)], linewidth=2, alpha=0.85)
    for r in results:
        color = LEVEL_COLORS.get(r.safety_level, '#95a5a6')
        P = r.center_3d
        ax2.scatter(*P, c=color, s=60, marker='s', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D — Wires & Hazards', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=-60)

    plt.suptitle('Power Line Corridor — Distance Measurement Report',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  组合可视化已保存: {output_path}")

    plt.close(fig)
    return output_path or ""


def print_report(results: List[DistanceResult],
                 uncertainties: Optional[List[Dict]] = None,
                 calibration_rmse: float = 0.0,
                 ) -> None:
    """打印测距报告"""
    print("\n" + "=" * 70)
    print("  POWER LINE CORRIDOR — DISTANCE MEASUREMENT REPORT")
    print("=" * 70)

    if calibration_rmse > 0:
        print(f"  Calibration RMSE: {calibration_rmse:.3f} m")

    print(f"\n  {'ID':>4s}  {'Class':>6s}  {'Distance':>10s}  "
          f"{'Uncertainty':>14s}  {'Level':>10s}  {'Closest Wire':>14s}")
    print("  " + "-" * 66)

    uncert_map = {}
    if uncertainties:
        uncert_map = {u['object_id']: u for u in uncertainties}

    for r in results:
        uncert = uncert_map.get(r.object_id)
        uncert_str = f"±{uncert['ci_95_half_width']:.2f}m" if uncert else "N/A"
        print(f"  {r.object_id:>4s}  {r.class_id:>6d}  {r.distance:>8.2f}m  "
              f"{uncert_str:>14s}  {r.safety_level:>10s}  {r.closest_wire_id:>14s}")

    # 统计
    dangers = [r for r in results if r.safety_level == "DANGER"]
    warnings = [r for r in results if r.safety_level == "WARNING"]
    print(f"\n  Summary: {len(dangers)} DANGER, {len(warnings)} WARNING, "
          f"{len(results) - len(dangers) - len(warnings)} SAFE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
#  交互式点击测距
# ═══════════════════════════════════════════════════════════════

def interactive_click_measure(
    image: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    metric_converter=None,
    calibration_rmse: float = 0.0,
    enable_uncertainty: bool = False,
    output_path: Optional[str] = None,
    title: str = "Click two points to measure 3D distance",
) -> List[TwoPointMeasurement]:
    """交互式点击测距工具

    用法：
      1. 运行此函数，弹出图像窗口
      2. 用鼠标点击图像上任意两个点
      3. 每次点击两点后，自动计算并显示 3D 距离
      4. 按 'u' 键撤销最近一次点击，按 'q' 键退出
      5. 关闭窗口后返回所有测量结果列表

    原理：
      点击 (u,v) → 邻域采样深度 → metric_converter 转米制
      → 针孔逆投影得 (Xc,Yc,Zc) → 欧氏距离

    Args:
        image: RGB 图像 (H, W, 3) 或图像路径
        depth_map: 深度图 (H, W)，可以是相对值或米制值
        intrinsics: 相机内参
        metric_converter: 深度→米制转换函数
        calibration_rmse: 标定误差，用于不确定性估计
        enable_uncertainty: 是否启用蒙特卡洛不确定性
        output_path: 保存带标注的结果图像路径
        title: 窗口标题

    Returns:
        [TwoPointMeasurement, ...]  每次成功测量的结果列表

    Example:
        >>> import cv2
        >>> from new_approach import Pipeline, interactive_click_measure
        >>> # 假设 pipeline 已运行，得到了 depth_metric 和 calibrator
        >>> results = interactive_click_measure(
        ...     pipeline.image,
        ...     pipeline.depth_metric,      # 或者 pipeline.depth_map + metric_converter
        ...     pipeline.intrinsics,
        ...     metric_converter=None,       # depth_metric 已是米制
        ...     output_path="click_measure_result.png",
        ... )
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    H, W = img.shape[:2]
    click_points: List[Tuple[float, float]] = []          # 当前这一对的点击点
    measurement_history: List[TwoPointMeasurement] = []    # 历史测量结果
    markers: List = []  # 用于重绘的标记元素

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

    # 状态文本
    status_text = ax.text(
        0.5, 0.02,
        "Click first point...",
        transform=ax.transAxes, fontsize=11,
        ha='center', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    def _redraw():
        """重绘图像上的所有标注"""
        # 移除旧标记
        for m in markers:
            try:
                m.remove()
            except Exception:
                pass
        markers.clear()

        # 绘制已点击的点
        for i, (u, v) in enumerate(click_points):
            color = '#3498db' if i == 0 else '#e74c3c'
            label = 'P1' if i == 0 else 'P2'
            # 十字标记
            line_h, = ax.plot([u - 12, u + 12], [v, v], color=color, linewidth=2)
            line_v, = ax.plot([u, u], [v - 12, v + 12], color=color, linewidth=2)
            dot, = ax.plot(u, v, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            text = ax.text(u + 14, v - 14, label, color='white', backgroundcolor=color,
                           fontsize=9, fontweight='bold', ha='left', va='top')
            markers.extend([line_h, line_v, dot, text])

        # 如果有两个点，绘制连线并计算距离
        if len(click_points) == 2:
            u1, v1 = click_points[0]
            u2, v2 = click_points[1]
            line, = ax.plot([u1, u2], [v1, v2], '--', color='#f39c12', linewidth=2, alpha=0.8)
            markers.append(line)

            # 计算3D距离
            conv = metric_converter
            if enable_uncertainty:
                result_unc = measure_two_points_with_uncertainty(
                    (u1, v1), (u2, v2), depth_map, intrinsics,
                    metric_converter=conv, calibration_rmse=calibration_rmse,
                )
                best = result_unc['best_estimate']
                if best:
                    measurement_history.append(best)
                    ci = result_unc['ci_95_half_width']
                    status_text.set_text(
                        f"P1({u1:.0f},{v1:.0f}) d={best.depth1_m:.1f}m  →  "
                        f"P2({u2:.0f},{v2:.0f}) d={best.depth2_m:.1f}m\n"
                        f"3D Distance: {best.distance_3d:.2f}m  ±{ci:.2f}m (95%CI)  |  "
                        f"Image: {best.distance_2d_image:.0f}px"
                    )
            else:
                best = measure_two_points(
                    (u1, v1), (u2, v2), depth_map, intrinsics,
                    metric_converter=conv,
                )
                if best:
                    measurement_history.append(best)
                    status_text.set_text(
                        f"P1({u1:.0f},{v1:.0f}) d={best.depth1_m:.1f}m  →  "
                        f"P2({u2:.0f},{v2:.0f}) d={best.depth2_m:.1f}m\n"
                        f"3D Distance: {best.distance_3d:.2f}m  |  "
                        f"Image: {best.distance_2d_image:.0f}px"
                    )

            # 连线中点标注距离
            mid_u, mid_v = (u1 + u2) / 2, (v1 + v2) / 2
            if measurement_history:
                dist_text = ax.text(
                    mid_u, mid_v - 10,
                    f"{measurement_history[-1].distance_3d:.2f}m",
                    color='white', backgroundcolor='#f39c12',
                    fontsize=11, fontweight='bold', ha='center', va='bottom',
                )
                markers.append(dist_text)

        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return

        # 如果已经有两个点，新点击重置
        if len(click_points) >= 2:
            click_points.clear()
            _redraw()

        u, v = event.xdata, event.ydata
        # 限制在图像范围内
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))
        click_points.append((u, v))

        if len(click_points) == 1:
            status_text.set_text(f"P1 selected ({u:.0f}, {v:.0f}) — click second point...")
        elif len(click_points) == 2:
            status_text.set_text("Computing 3D distance...")

        _redraw()
        print(f"  Click: ({u:.0f}, {v:.0f})")

    def _on_key(event):
        if event.key == 'q':
            plt.close(fig)
        elif event.key == 'u':
            if click_points:
                removed = click_points.pop()
                print(f"  Undo: ({removed[0]:.0f}, {removed[1]:.0f})")
                status_text.set_text(
                    "Click first point..." if not click_points
                    else f"P1 selected — click second point..."
                )
                _redraw()
        elif event.key == 'r':
            click_points.clear()
            markers.clear()
            status_text.set_text("Reset. Click first point...")
            _redraw()
        elif event.key == 'h':
            # 打印历史记录
            print(f"\n  === Measurement History ({len(measurement_history)} pairs) ===")
            for i, m in enumerate(measurement_history):
                uncert = f" ±{m.uncertainty_95:.2f}m" if m.uncertainty_95 else ""
                print(f"  [{i+1}] {m.distance_3d:.2f}m{uncert}  "
                      f"P1({m.point1_2d[0]:.0f},{m.point1_2d[1]:.0f}) → "
                      f"P2({m.point2_2d[0]:.0f},{m.point2_2d[1]:.0f})")
            print("  =" + "=" * 50)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    # 使用说明
    instructions = (
        "Instructions:\n"
        "  Click: select point\n"
        "  U: undo last click\n"
        "  R: reset current pair\n"
        "  H: print history\n"
        "  Q: quit"
    )
    ax.text(0.99, 0.99, instructions, transform=ax.transAxes,
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 保存结果图
    if output_path and measurement_history:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # 重新画一张最终的图
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        ax2.imshow(img)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(measurement_history)))
        for i, m in enumerate(measurement_history):
            u1, v1 = m.point1_2d
            u2, v2 = m.point2_2d
            c = colors[i]
            ax2.plot([u1, u2], [v1, v2], '-', color=c, linewidth=2)
            ax2.plot(u1, v1, 'o', color=c, markersize=6)
            ax2.plot(u2, v2, 's', color=c, markersize=6)
            mid_u, mid_v = (u1 + u2) / 2, (v1 + v2) / 2
            ax2.text(mid_u, mid_v, f"{m.distance_3d:.2f}m",
                     color='white', backgroundcolor='black', fontsize=8, ha='center')
        ax2.set_title(f"Click-to-Measure Results ({len(measurement_history)} measurements)",
                      fontsize=13, fontweight='bold')
        ax2.axis('off')
        fig2.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"\n  交互式测距结果已保存: {output_path}")

    # 打印汇总
    if measurement_history:
        print(f"\n  === Click-to-Measure Summary ===")
        print(f"  Total measurements: {len(measurement_history)}")
        dists = [m.distance_3d for m in measurement_history]
        print(f"  Min: {min(dists):.2f}m  Max: {max(dists):.2f}m  Mean: {np.mean(dists):.2f}m")
        print(f"  ==" + "=" * 30)

    return measurement_history
