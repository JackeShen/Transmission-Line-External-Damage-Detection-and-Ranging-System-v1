"""
一键运行：点击图像两点测距
===========================
用法：在终端里运行这个脚本，弹出图像窗口，鼠标点两点，自动算3D距离。

最简用法（只需3样东西）：
    python run_click_measure.py --image 图片.jpg --depth 深度图.npy

"""

import argparse
import sys
import os
import numpy as np
import cv2

# 把父目录加到 path，方便直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_approach.geometric_utils import CameraIntrinsics
from new_approach.visualizer import interactive_click_measure


def main():
    parser = argparse.ArgumentParser(
        description="交互式两点测距 —— 点击图像上任意两点，计算3D距离"
    )
    # ── 必填参数 ──
    parser.add_argument("--image", required=True,
                        help="原始航拍图像路径 (jpg/png)")
    parser.add_argument("--depth", required=True,
                        help="深度图路径 (.npy文件)")

    # ── 相机内参（可选，有默认值）──
    parser.add_argument("--fx", type=float, default=3714.81,
                        help="x方向焦距 (默认: 3714.81)")
    parser.add_argument("--fy", type=float, default=3714.81,
                        help="y方向焦距 (默认: 3714.81)")
    parser.add_argument("--cx", type=float, default=-9.96,
                        help="主点x坐标 (默认: -9.96)")
    parser.add_argument("--cy", type=float, default=-37.70,
                        help="主点y坐标 (默认: -37.70)")

    # ── 深度标定（可选）──
    parser.add_argument("--scale_factor", type=float, default=None,
                        help="深度→米制的比例因子。不填则假设深度图已经是米制。"
                             "例如 depth_metric = raw_depth * scale_factor")
    parser.add_argument("--depth_is_metric", action="store_true",
                        help="如果深度图已经是米制单位，加这个参数")

    # ── 输出 ──
    parser.add_argument("--output", type=str, default="click_measure_result.png",
                        help="结果图保存路径 (默认: click_measure_result.png)")
    parser.add_argument("--uncertainty", action="store_true",
                        help="启用蒙特卡洛不确定性估计（较慢）")

    args = parser.parse_args()

    # ═══════════════════════════════════════
    # Step 1: 加载图像
    # ═══════════════════════════════════════
    print("=" * 60)
    print("  交互式两点测距工具")
    print("=" * 60)
    print(f"\n[1/3] 加载图像: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"  ❌ 无法加载图像: {args.image}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  分辨率: {img.shape[1]}×{img.shape[0]}")

    # ═══════════════════════════════════════
    # Step 2: 加载深度图
    # ═══════════════════════════════════════
    print(f"\n[2/3] 加载深度图: {args.depth}")
    depth = np.load(args.depth)
    print(f"  形状: {depth.shape}, 范围: [{depth.min():.3f}, {depth.max():.3f}]")

    # 确定米制转换方式
    if args.depth_is_metric:
        metric_converter = None
        print("  深度图已为米制，不需要 scale_factor")
    elif args.scale_factor is not None:
        metric_converter = lambda d: d * args.scale_factor
        print(f"  使用 scale_factor = {args.scale_factor}")
        print(f"  即: metric_depth = raw_depth × {args.scale_factor}")
    else:
        # 没有给 scale_factor，假设深度已经是米制，给出提醒
        print("  ⚠ 未指定 --scale_factor 或 --depth_is_metric")
        print("  假设深度图已经是米制单位。如果不对，请加 --scale_factor <数值>")
        metric_converter = None

    # ═══════════════════════════════════════
    # Step 3: 设置相机内参
    # ═══════════════════════════════════════
    print(f"\n[3/3] 相机内参: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    intrinsics = CameraIntrinsics(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        width=img.shape[1], height=img.shape[0],
    )

    # ═══════════════════════════════════════
    # 启动交互式测量
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  开始交互式测量")
    print("=" * 60)
    print("""
  操作说明:
    🖱️  鼠标点击图像上任意两个点  →  自动计算3D距离
    ⌨️   U  = 撤销上一次点击
    ⌨️   R  = 重置当前这一对
    ⌨️   H  = 打印所有历史测量记录
    ⌨️   Q  = 退出

  窗口弹出后，直接点两个你想测距的位置即可！
  ═══════════════════════════════════════════════════
""")

    results = interactive_click_measure(
        image=img,
        depth_map=depth,
        intrinsics=intrinsics,
        metric_converter=metric_converter,
        calibration_rmse=0.0,
        enable_uncertainty=args.uncertainty,
        output_path=args.output,
        title="点击任意两点 → 计算3D距离 | Q=退出 U=撤销 H=历史",
    )

    # ═══════════════════════════════════════
    # 打印最终结果
    # ═══════════════════════════════════════
    if results:
        print(f"\n✅ 共完成 {len(results)} 次测量\n")
        for i, r in enumerate(results):
            uncert = f" ±{r.uncertainty_95:.2f}m" if r.uncertainty_95 else ""
            print(f"  [{i+1}] {r.distance_3d:.2f}m{uncert}")
            print(f"       P1({r.point1_2d[0]:.0f},{r.point1_2d[1]:.0f}) "
                  f"深度={r.depth1_m:.1f}m")
            print(f"       P2({r.point2_2d[0]:.0f},{r.point2_2d[1]:.0f}) "
                  f"深度={r.depth2_m:.1f}m")
    else:
        print("\n⚠ 未完成任何测量（可能窗口被直接关闭了）")


if __name__ == "__main__":
    main()
