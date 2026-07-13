#!/usr/bin/env python3
"""
depth_to_3d.py — 从深度图和相机内参生成三维点云

基于针孔相机模型，将深度图的每个像素反投影到相机坐标系，构建三维点云。

┌─────────────────────────────────────────────────────┐
│  针孔模型逆投影：                                     │
│    Xc = (u - cx) * Zc / fx                          │
│    Yc = (v - cy) * Zc / fy                          │
│    Zc = depth_value                                 │
│                                                     │
│  坐标系：相机坐标系（右手系）                          │
│    Xc → 右    Yc → 下    Zc → 前（深度方向）           │
└─────────────────────────────────────────────────────┘

输入：
  --depth          深度图 .npy 文件（必须）
  --image          原始RGB航拍图像（可选，用于点云着色）
  --fx/--fy/--cx/--cy  相机内参
  --scale          深度→米制比例因子
  --step           下采样步长（控制点云密度）

输出：
  - *_pointcloud.ply     PLY格式3D点云（MeshLab / CloudCompare 可打开）
  - *_scene_3d.png       3D可视化图（三个视角：侧视/俯视/透视）


用法示例：
  # 最简用法（使用默认内参，假设深度已是米制）
  python depth_to_3d.py --depth depth_map.npy

  # 指定自定义内参 + RGB着色
  python depth_to_3d.py --depth depth.npy --image DJI_001.jpg \\
      --fx 3714.81 --fy 3714.81 --cx -9.96 --cy -37.70

  # 相对深度 + 比例因子 + 限制最大深度
  python depth_to_3d.py --depth depth.npy --scale 0.001 --max-depth 200

  # 高清输出
  python depth_to_3d.py --depth depth.npy --image img.jpg --step 2 --dpi 300
"""

import argparse, os, sys, time
import numpy as np
from typing import Optional, Tuple

# ── 可选依赖 ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ═══════════════════════════════════════════════════════════
#  核心：深度图 → 3D点云
# ═══════════════════════════════════════════════════════════

def depth_map_to_point_cloud(
    depth_metric: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: Optional[np.ndarray] = None,
    step: int = 8,
    max_depth: Optional[float] = None,
    max_points: int = 2_000_000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """将米制深度图反投影为相机坐标系下的3D点云。

    对每个有效像素 (u, v)，执行针孔逆投影：

        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        Zc = depth_metric[v, u]

    Args:
        depth_metric:  米制深度图 (H, W)，单位：米。无效区域应 ≤ 0。
        fx, fy:        焦距（像素单位）。
        cx, cy:        主点坐标（像素单位）。
        rgb_image:     可选 RGB 图像 (H, W, 3)，用于给点云着色。
        step:          采样步长，每隔 step 个像素取一个点。
        max_depth:     最大深度阈值（米），超过此值的点丢弃。
        max_points:    最大输出点数，超过时自动增大步长。

    Returns:
        points:  (N, 3) float64  相机坐标系 3D 点 [Xc, Yc, Zc]。
        colors:  (N, 3) float32  归一化 RGB 颜色 [0,1]，无图像时为 None。
    """
    H, W = depth_metric.shape

    # ── 自动调整步长以防内存爆炸 ──
    est_points = (H // step) * (W // step)
    if est_points > max_points:
        step = max(step, int(np.ceil(np.sqrt(H * W / max_points))))
        print(f"  ⚠ 预估点数超限，自动增大 step → {step}")

    # ── 生成下采样网格 ──
    v_idx = np.arange(0, H, step)
    u_idx = np.arange(0, W, step)
    uu, vv = np.meshgrid(u_idx, v_idx)  # (h, w)

    Zc = depth_metric[vv, uu].astype(np.float64)

    # ── 有效性掩码 ──
    valid = Zc > 0
    if max_depth is not None:
        valid &= Zc <= max_depth

    if not np.any(valid):
        return np.empty((0, 3)), None

    u_val = uu[valid]
    v_val = vv[valid]
    Z_val = Zc[valid]

    # ── 针孔逆投影（向量化） ──
    Xc = (u_val - cx) * Z_val / fx
    Yc = (v_val - cy) * Z_val / fy

    points = np.column_stack([Xc, Yc, Z_val])

    # ── 颜色提取 ──
    colors = None
    if rgb_image is not None and rgb_image.ndim == 3:
        c = rgb_image[v_val, u_val].astype(np.float32) / 255.0
        colors = np.clip(c, 0.0, 1.0)

    return points, colors


# ═══════════════════════════════════════════════════════════
#  输出：PLY 文件
# ═══════════════════════════════════════════════════════════

def save_ply(filepath: str,
             points: np.ndarray,
             colors: Optional[np.ndarray] = None) -> int:
    """保存为 ASCII PLY 格式（通用三维点云格式）。

    Args:
        filepath: 输出 .ply 路径。
        points:   (N, 3) 点坐标。
        colors:   (N, 3) RGB [0,1] 范围。

    Returns:
        写入的点数。
    """
    n = len(points)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is not None:
            c_uint = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            for i in range(n):
                f.write(
                    f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f} "
                    f"{c_uint[i, 0]} {c_uint[i, 1]} {c_uint[i, 2]}\n"
                )
        else:
            for i in range(n):
                f.write(f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f}\n")

    file_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  PLY 已保存: {filepath}  ({n:,} 点, {file_mb:.1f} MB)")
    return n


# ═══════════════════════════════════════════════════════════
#  输出：3D 可视化图
# ═══════════════════════════════════════════════════════════

def _random_sample(arr: np.ndarray, n: int) -> np.ndarray:
    """对数组第一维随机采样"""
    if len(arr) <= n:
        return arr
    idx = np.random.choice(len(arr), n, replace=False)
    return arr[idx]


def _pseudo_color(z: np.ndarray) -> np.ndarray:
    """用深度值生成伪彩色（不依赖 matplotlib）"""
    z_clip = np.clip(z, 0, None)
    z_norm = (z_clip - z_clip.min()) / (z_clip.max() - z_clip.min() + 1e-8)

    # 简单的蓝→青→绿→黄→红 colormap
    r = np.clip((z_norm - 0.5) * 2.0, 0, 1)
    g = np.clip(1.0 - np.abs(z_norm - 0.5) * 2.0, 0, 1)
    b = np.clip((0.5 - z_norm) * 2.0, 0, 1)

    return np.column_stack([r, g, b])


def visualize_3d(points: np.ndarray,
                 colors: Optional[np.ndarray] = None,
                 output_path: str = "scene_3d.png",
                 elev: float = 25, azim: float = -60,
                 title: str = "3D Point Cloud from Depth Map",
                 dpi: int = 200,
                 viz_max_points: int = 150_000) -> None:
    """生成多视角 3D 可视化图（1 行 3 列：侧视 / 俯视 / 透视）。

    Args:
        points:          (N, 3) 相机坐标点。
        colors:          (N, 3) RGB [0,1]。
        output_path:     输出图片路径。
        elev, azim:      透视视角仰角/方位角。
        title:           总标题。
        dpi:             输出分辨率。
        viz_max_points:  可视化最大点数（性能限制）。
    """
    if not HAS_MPL:
        print("  ⚠ matplotlib 未安装，跳过多视角可视化。")
        return

    # 降采样到可绘制的数量
    if len(points) > viz_max_points:
        idx = np.random.choice(len(points), viz_max_points, replace=False)
        pts = points[idx]
        cols = colors[idx] if colors is not None else None
    else:
        pts = points
        cols = colors

    # 配色
    if cols is None:
        cols = _pseudo_color(pts[:, 2])  # 按深度着色

    # ── 建立三个视角 ──
    views = [
        ("透视", elev, azim),
        ("侧视 (X-Z)", 0, -90),
        ("俯视 (X-Z)", 90, -90),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7),
                              subplot_kw={'projection': '3d'})
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)

    for ax, (vname, el, az) in zip(axes, views):
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=cols, s=0.4, alpha=0.75, marker=".",
            rasterized=True,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(vname, fontsize=12, fontweight="bold")
        ax.view_init(elev=el, azim=az)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  3D 可视化已保存: {output_path}")


# ═══════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="深度图 → 3D点云：基于针孔相机模型反投影生成三维场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法（默认大疆L1内参）
  python depth_to_3d.py --depth depth_map.npy

  # RGB 着色 + 高分辨率
  python depth_to_3d.py --depth depth.npy --image DJI_001.jpg --step 4

  # 相对深度需转换 + 限制深度范围
  python depth_to_3d.py --depth depth.npy --scale 0.001 --max-depth 150

  # 输出到指定目录
  python depth_to_3d.py --depth depth.npy --image img.jpg -o ./results

  # 自定义相机内参
  python depth_to_3d.py --depth depth.npy --fx 2500 --fy 2500 --cx 960 --cy 540
        """,
    )

    # ── 输入 ──
    p.add_argument("--depth", required=True,
                   help="深度图 .npy 文件（必须）")
    p.add_argument("--image", default=None,
                   help="RGB 航拍原图（可选，用于给点云着色）")

    # ── 相机内参 ──
    p.add_argument("--fx", type=float, default=3714.81,
                   help="x 方向焦距 px（默认 3714.81）")
    p.add_argument("--fy", type=float, default=3714.81,
                   help="y 方向焦距 px（默认 3714.81）")
    p.add_argument("--cx", type=float, default=-9.96,
                   help="主点 x px（默认 -9.96）")
    p.add_argument("--cy", type=float, default=-37.70,
                   help="主点 y px（默认 -37.70）")

    # ── 深度转换 ──
    p.add_argument("--scale", type=float, default=None,
                   help="深度→米制比例因子，metric = raw * scale")
    p.add_argument("--depth-is-metric", action="store_true",
                   help="深度图已是米制，跳过转换")
    p.add_argument("--max-depth", type=float, default=None,
                   help="最大深度阈值（米），超出丢弃")

    # ── 生成参数 ──
    p.add_argument("--step", type=int, default=4,
                   help="采样步长，越大点越少、越快（默认 4）")
    p.add_argument("--max-points", type=int, default=3_000_000,
                   help="最大点数上限，超出自动增大步长（默认 300万）")

    # ── 输出 ──
    p.add_argument("-o", "--output", type=str, default=".",
                   help="输出目录（默认当前目录）")
    p.add_argument("--prefix", type=str, default=None,
                   help="输出文件名前缀（默认从深度文件名推断）")
    p.add_argument("--no-ply", action="store_true",
                   help="不导出 PLY 文件")
    p.add_argument("--no-viz", action="store_true",
                   help="不生成可视化图")
    p.add_argument("--dpi", type=int, default=200,
                   help="可视化图 DPI（默认 200）")
    p.add_argument("--title", type=str,
                   default="3D Point Cloud from Depth Map",
                   help="可视化图标题")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子，保证可视化降采样的可重复性（默认 42）")

    return p.parse_args()


def main():
    args = _parse_args()
    np.random.seed(args.seed)
    t_start = time.time()

    print("=" * 64)
    print("  深 度 图  →  三 维 点 云")
    print("  针孔相机模型反投影重建")
    print("=" * 64)

    # ── Step 1: 加载深度图 ──
    print(f"\n[1/4] 加载深度图: {args.depth}")
    depth_raw = np.load(args.depth)
    H, W = depth_raw.shape
    n_valid = int(np.sum(depth_raw > 0))
    print(f"  分辨率    : {W} × {H}  ({depth_raw.size / 1e6:.1f} Mpx)")
    print(f"  值域      : [{depth_raw[depth_raw > 0].min():.4f}, "
          f"{depth_raw.max():.4f}]")
    print(f"  有效像素  : {n_valid:,} / {depth_raw.size:,}  "
          f"({100 * n_valid / depth_raw.size:.1f}%)")

    # ── Step 2: 深度预处理 ──
    print(f"\n[2/4] 深度预处理")
    if args.depth_is_metric:
        depth_metric = depth_raw.astype(np.float64)
        print("  模式: 已是米制，不转换")
    elif args.scale is not None:
        depth_metric = depth_raw.astype(np.float64) * args.scale
        mask = depth_metric > 0
        print(f"  模式: 比例因子  scale = {args.scale}")
        print(f"  转换后范围: [{depth_metric[mask].min():.2f}, "
              f"{depth_metric[mask].max():.2f}] m")
    else:
        print("  ⚠ 未指定 --scale 或 --depth-is-metric，")
        print("    假设深度图已是米制。若不对请加 --scale <值>")
        depth_metric = depth_raw.astype(np.float64)

    if args.max_depth:
        print(f"  最大深度阈值: {args.max_depth} m")

    # ── Step 3: 加载 RGB 图像 ──
    rgb = None
    if args.image:
        print(f"\n[3/4] 加载 RGB 图像: {args.image}")
        if HAS_CV2:
            bgr = cv2.imread(args.image)
            if bgr is None:
                print(f"  ⚠ 无法读取: {args.image}")
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (H, W):
                    print(f"  ⚠ 图像尺寸 {rgb.shape[1]}×{rgb.shape[0]} "
                          f"与深度图 {W}×{H} 不一致，将自动缩放")
                    rgb = cv2.resize(rgb, (W, H))
                print(f"  分辨率: {rgb.shape[1]} × {rgb.shape[0]}")
        else:
            print("  ⚠ OpenCV 未安装，跳过 RGB 着色。pip install opencv-python")

    # ── Step 4: 生成 3D 点云 ──
    print(f"\n[4/4] 生成 3D 点云")
    print(f"  内参:  fx={args.fx:.2f}  fy={args.fy:.2f}  "
          f"cx={args.cx:.2f}  cy={args.cy:.2f}")
    print(f"  步长:  step={args.step}")

    points, colors = depth_map_to_point_cloud(
        depth_metric,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        rgb_image=rgb,
        step=args.step,
        max_depth=args.max_depth,
        max_points=args.max_points,
    )

    if len(points) == 0:
        print("\n  ❌ 没有有效 3D 点！")
        print("     可能原因: ①深度值全为0或负数 ② max_depth 过小 ③ 比例因子不对")
        sys.exit(1)

    # ── 打印统计信息 ──
    extent_x = points[:, 0].max() - points[:, 0].min()
    extent_y = points[:, 1].max() - points[:, 1].min()
    extent_z = points[:, 2].max() - points[:, 2].min()

    print(f"\n  ─── 3D 点云统计 ───")
    print(f"  总点数      : {len(points):,}")
    print(f"  X (右→)     : [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] m  "
          f"跨度 {extent_x:.1f} m")
    print(f"  Y (下→)     : [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] m  "
          f"跨度 {extent_y:.1f} m")
    print(f"  Z (前→)     : [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] m  "
          f"跨度 {extent_z:.1f} m")
    print(f"  平均深度    : {points[:, 2].mean():.1f} m")
    print(f"  中位深度    : {np.median(points[:, 2]):.1f} m")
    print(f"  着色        : {'RGB 原图' if colors is not None else '深度伪彩'}")

    # ── 保存输出 ──
    os.makedirs(args.output, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.depth))[0]

    if not args.no_ply:
        save_ply(
            os.path.join(args.output, f"{prefix}_pointcloud.ply"),
            points, colors,
        )

    if not args.no_viz:
        visualize_3d(
            points, colors,
            output_path=os.path.join(args.output, f"{prefix}_scene_3d.png"),
            title=args.title,
            dpi=args.dpi,
        )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"  完成! 耗时 {elapsed:.1f}s  |  输出: {os.path.abspath(args.output)}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
