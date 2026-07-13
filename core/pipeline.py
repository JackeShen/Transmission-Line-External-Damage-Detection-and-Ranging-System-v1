#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户数据处理管道 - 危险物与输电线距离分析

比例因子调整说明：
1. 比例因子定义：将深度图中的相对深度值转换为实际物理距离的系数
2. 距离计算公式：实际距离（米）= 比例因子 / 深度值

比例因子设置方法：
1. 手动设置：通过scale_factor参数直接指定一个固定值
   - 默认值为20.0
   - 根据场景和相机参数可能需要调整

2. 基于标定物自动计算（推荐）：
   - 指定一个在图像中已知实际距离的物体作为标定物
   - 设置calibration_class_id（标定物的类别ID）
   - 设置calibration_distance（标定物到摄像头的实际距离，单位：米）
   - 系统会自动计算：scale_factor = 实际距离 * 深度值

3. 使用默认值：如果未指定任何参数，将使用默认比例因子20.0

标定物使用建议：
- 选择图像中清晰可见、特征明显的物体
- 确保已知该物体到摄像头的准确距离
- 对于最佳精度，选择与目标分析区域距离相近的标定物
- 图像中可以有多个相同类别的标定物，系统会计算它们的平均值

示例：
如果图像中有一个类别ID为0的物体，已知其距离摄像头1000米，
则系统会提取该物体区域的深度值，然后计算比例因子：
scale_factor = 1000 * 深度值
之后使用此比例因子计算其他物体的实际距离。
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from core.spatial_analyzer import SpatialAnalyzer
import cv2

def calculate_scale_factor_from_calibration(depth_map, danger_objects, calibration_class_id, calibration_distance):
    """
    根据标定物计算比例因子
    
    Args:
        depth_map: 深度图
        danger_objects: 危险物信息字典
        calibration_class_id: 标定物的类别ID
        calibration_distance: 标定物到摄像头的实际距离（米）
        
    Returns:
        float: 计算得到的比例因子
    """
    print(f"\n正在使用标定物计算比例因子...")
    print(f"  标定物类别ID: {calibration_class_id}")
    print(f"  标定物实际距离: {calibration_distance} 米")
    
    # 查找标定物类别
    calibration_objects = []
    for obj_id, obj in danger_objects.items():
        if obj['class_id'] == calibration_class_id:
            calibration_objects.append((obj_id, obj))
    
    if not calibration_objects:
        print(f"警告: 在图像中未找到类别ID为 {calibration_class_id} 的标定物")
        return None
    
    print(f"找到 {len(calibration_objects)} 个标定物实例")
    print(f"  仅使用第一个标定物实例 (#{calibration_objects[0][0]}) 进行计算")
    
    # 只使用第一个标定物
    obj_id, obj = calibration_objects[0]
    bbox = obj['bbox']
    x_min, y_min, x_max, y_max = bbox
    
    # 提取标定物区域的深度值
    region_depth = depth_map[y_min:y_max, x_min:x_max]
    
    # 过滤掉无效深度值（如果有的话）
    valid_depths = region_depth[region_depth > 0]
    
    if len(valid_depths) == 0:
        print(f"警告: 无法从标定物 #{obj_id} 区域获取有效深度值")
        return None
    
    # 计算第一个标定物的平均深度
    final_avg_depth = np.mean(valid_depths)
    print(f"  标定物 #{obj_id} 区域平均深度: {final_avg_depth:.4f}")
    
    # 使用公式计算比例因子: scale_factor = 实际距离 / 深度值
    # 这样实际距离 = 深度值 * scale_factor (成正比关系)
    scale_factor = calibration_distance / final_avg_depth
    print(f"  计算得到的比例因子: {scale_factor:.2f}")
    
    return scale_factor

def prepare_user_data(image_path, annotation_path, segmentation_path, depth_map_path=None, scale_factor=None, 
                     calibration_class_id=None, calibration_distance=None, camera_matrix=None, 
                     fx=None, fy=None, cx=None, cy=None, regions_json_path=None):
    """
    准备用户数据，确保格式正确
    
    Args:
        image_path: 用户图像路径
        annotation_path: YOLO格式的txt标注文件路径
        segmentation_path: JSON格式的分割标注文件路径
        depth_map_path: 可选的深度图npy文件路径
        scale_factor: 可选的比例因子
        calibration_class_id: 可选的标定物类别ID
        calibration_distance: 可选的标定物实际距离（米）
        camera_matrix: 可选的3x3相机内参矩阵
        fx: 可选的焦距x方向
        fy: 可选的焦距y方向
        cx: 可选的主点x坐标
        cy: 可选的主点y坐标
        
    Returns:
        tuple: (depth_map, danger_objects, segmentation_path, scale_factor, camera_params)
        其中camera_params是包含相机内参的字典
    """
    print("开始准备用户数据...")
    
    # 1. 加载图像获取尺寸信息
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    image_height, image_width = image.shape[:2]
    print(f"图像加载成功: {image_path} ({image_width}x{image_height})")
    
    # 2. 加载YOLO格式的标注文件
    danger_objects = load_yolo_annotations(annotation_path, image_width, image_height)
    print(f"YOLO标注加载成功: {annotation_path} ({len(danger_objects)}个危险物)")
    
    # 3. 检查分割标注文件
    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(f"无法找到分割标注文件: {segmentation_path}")
    print(f"分割标注文件存在: {segmentation_path}")
    
    # 4. 加载深度图
    if depth_map_path and os.path.exists(depth_map_path):
        depth_map = np.load(depth_map_path)
        print(f"深度图加载成功: {depth_map_path}")
    else:
        raise FileNotFoundError(f"无法找到深度图文件: {depth_map_path}")
    
    # 5. 确定比例因子
    if calibration_class_id is not None and calibration_distance is not None:
        # 使用标定物计算比例因子
        calculated_scale = calculate_scale_factor_from_calibration(
            depth_map, danger_objects, calibration_class_id, calibration_distance)
        if calculated_scale is not None:
            scale_factor = calculated_scale
        else:
            # 如果标定失败，使用提供的比例因子或默认值
            if scale_factor is None:
                scale_factor = 20.0  # 默认比例因子
            print(f"标定失败，使用比例因子: {scale_factor}")
    elif scale_factor is None:
        # 如果没有提供比例因子和标定信息，使用默认值
        scale_factor = 20.0  # 默认比例因子
        print(f"使用默认比例因子: {scale_factor}")
    else:
        print(f"使用提供的比例因子: {scale_factor}")
    
    # 将相机内参打包成字典返回
    camera_params = {
        'camera_matrix': camera_matrix,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }
    
    return depth_map, danger_objects, segmentation_path, scale_factor, camera_params

def load_yolo_annotations(annotation_path, image_width, image_height):
    """
    加载YOLO格式的标注文件并转换为需要的格式
    
    YOLO格式: 每行包含 class_id center_x center_y width height (归一化坐标)
    
    Args:
        annotation_path: YOLO格式的txt标注文件路径
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        dict: 危险物信息字典
    """
    danger_objects = {}
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"警告: 跳过无效的标注行 {idx+1}: {line.strip()}")
            continue
        
        try:
            class_id = int(parts[0])
            # YOLO使用归一化坐标，需要转换为像素坐标
            center_x = float(parts[1]) * image_width
            center_y = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            
            # 计算边界框坐标 [x_min, y_min, x_max, y_max]
            x_min = int(center_x - width / 2)
            y_min = int(center_y - height / 2)
            x_max = int(center_x + width / 2)
            y_max = int(center_y + height / 2)
            
            # 确保坐标在有效范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width - 1, x_max)
            y_max = min(image_height - 1, y_max)
            
            # 添加到危险物字典
            danger_objects[str(idx)] = {
                'class_id': class_id,
                'bbox': [x_min, y_min, x_max, y_max],
                'estimated_distance': None  # 将在后续计算
            }
        except Exception as e:
            print(f"警告: 处理标注行 {idx+1} 时出错: {e}")
    
    return danger_objects



def calculate_distances(depth_map, danger_objects, segmentation_path, scale_factor, camera_matrix=None, fx=None, fy=None, cx=None, cy=None, regions_json_path=None, enable_smoothing=True, enable_y_factor_adjustment=True, calibration_class_id=None, calibration_distance=None, distance_method='segment'):
    """
    使用SpatialAnalyzer计算危险物与输电线之间的距离
    
    Args:
        depth_map: 深度图
        danger_objects: 危险物信息字典
        segmentation_path: 分割标注文件路径
        scale_factor: 比例因子
        camera_matrix: 3x3相机内参矩阵（可选）
        fx: 焦距x方向（可选，若未提供camera_matrix时使用）
        fy: 焦距y方向（可选，若未提供camera_matrix时使用）
        cx: 主点x坐标（可选，若未提供camera_matrix时使用）
        cy: 主点y坐标（可选，若未提供camera_matrix时使用）
        regions_json_path: 区域划分JSON文件路径（可选，用于区域化深度平滑）
        enable_smoothing: 是否启用深度平滑（可选，默认True）
        enable_y_factor_adjustment: 是否启用基于图像垂直位置的深度调整，使图像上方的点深度增加（可选，默认True）
        calibration_class_id: 标定物的类别ID（可选）
        calibration_distance: 标定物到摄像头的实际距离（米）（可选）
        
    Returns:
        tuple: (analyzer实例, 距离计算结果)
    """
    print("初始化空间分析器...")
    analyzer = SpatialAnalyzer()
    
    # 设置数据
    analyzer.set_depth_map(depth_map)
    analyzer.set_scale_factor(scale_factor)
    analyzer.add_danger_objects(danger_objects)
    
    # 设置标定物信息（如果提供）
    if calibration_class_id is not None:
        # 查找标定物
        for obj_id, obj in danger_objects.items():
            if obj['class_id'] == calibration_class_id:
                print(f"找到标定物 (ID: {obj_id}, 类别: {calibration_class_id})")
                analyzer.set_calibration_object(obj)
                break
    
    # 设置相机内参（如果提供）
    if camera_matrix is not None:
        print(f"设置相机内参矩阵")
        analyzer.set_camera_intrinsics(camera_matrix=camera_matrix)
    elif fx is not None and fy is not None and cx is not None and cy is not None:
        print(f"设置相机内参参数: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        analyzer.set_camera_intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    else:
        print("未提供相机内参，将使用默认的近似值")
    
    # 加载输电线分割标注
    print(f"加载输电线分割标注: {segmentation_path}")
    success = analyzer.load_power_line_annotations(segmentation_path)
    if not success:
        raise Exception("加载输电线标注失败")
    
    # 平滑输电线深度，实现由近及远的效果
    if enable_smoothing:
        print("平滑输电线深度，实现深度渐变效果...")
        if regions_json_path:
            print(f"使用区域化深度平滑，区域文件: {regions_json_path}")
        analyzer.smooth_power_line_depths(apply_global_smooth=True, 
                                         global_sigma=1.0, 
                                         line_direction_smooth=True, 
                                         depth_gradient_factor=0.3,
                                         regions_json_path=regions_json_path,
                                         enable_y_factor_adjustment=enable_y_factor_adjustment)
    else:
        print("输电线深度平滑功能已禁用")
    
    # 构建三维空间
    print("构建三维空间...")
    success = analyzer.build_3d_space()
    if not success:
        raise Exception("构建三维空间失败")
    
    # 计算距离
    print("计算距离...")
    results = analyzer.calculate_all_distances(distance_method=distance_method)
    
    return analyzer, results

def save_results(results, output_dir="user_results"):
    """
    保存距离计算结果
    
    Args:
        results: 距离计算结果
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存距离结果
    output_file = os.path.join(output_dir, "distance_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"距离计算结果已保存到: {output_file}")
    return output_file

def visualize_with_user_data(image_path, analyzer, results, output_dir="user_results", calibration_class_id=None, calibration_distance=None, scale_factor=None, show_all_points_in_3d=False, max_all_points=10000, draw_distance_from='center'):
    """
    Visualize analysis results with user data, including shortest distance lines between danger objects and power lines
    
    Args:
        image_path: Original image path
        analyzer: SpatialAnalyzer instance
        results: Distance calculation results
        output_dir: Output directory
        calibration_class_id: 标定物的类别ID（可选）
        calibration_distance: 标定物到摄像头的实际距离（米）（可选）
        draw_distance_from: 距离线段的绘制起点，可选值：'center'（危险物中心）或'selected'（选择的点），默认'center'
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        # Draw power lines first (ensure they are in the background)
        for line in analyzer.power_lines:
            if 'points' in line:
                points = np.array(line['points'])
                ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=2)
                
                # Mark power line ID
                mid_idx = len(points) // 2
                ax.text(points[mid_idx, 0], points[mid_idx, 1], 
                       line.get('id', ''), color='white', backgroundcolor='green')
        
        # Find all calibration objects first
        calibration_objects_list = []
        for obj in analyzer.danger_objects:
            if calibration_class_id is not None and 'class_id' in obj and obj['class_id'] == calibration_class_id:
                calibration_objects_list.append(obj['id'])
        
        # Draw danger objects with bounding boxes, distances, and shortest distance lines
        for obj in analyzer.danger_objects:
            bbox = obj['bbox']
            
            # Check if it's the calibration object or same category as calibration object
            is_calibration_object = (calibration_class_id is not None and 
                                    'class_id' in obj and 
                                    obj['class_id'] == calibration_class_id and
                                    obj['id'] == calibration_objects_list[0] if calibration_objects_list else False)
            
            is_same_category_as_calibration = (calibration_class_id is not None and 
                                               'class_id' in obj and 
                                               obj['class_id'] == calibration_class_id and
                                               not is_calibration_object)
            
            # Use blue for calibration object, yellow for same category, red for others
            if is_calibration_object:
                edgecolor = 'blue'
            elif is_same_category_as_calibration:
                edgecolor = 'yellow'
            else:
                edgecolor = 'red'
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                fill=False, edgecolor=edgecolor, linewidth=2, alpha=1.0)
            ax.add_patch(rect)
            
            # Draw center point
            # 直接从bbox计算中心点，确保始终是边界框的几何中心
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            # 同时保存原始center_2d的值，用于比较
            original_center_x, original_center_y = obj['center_2d']
            # Use blue for calibration object, yellow for same category, red for others
            if is_calibration_object:
                marker_color = 'bo'
            elif is_same_category_as_calibration:
                marker_color = 'yo'
            else:
                marker_color = 'ro'
            # 绘制边界框的几何中心
            ax.plot(bbox_center_x, bbox_center_y, marker_color, markersize=5)
            
            # 如果center_2d被修改为最近点，绘制一个小的绿色点来标记
            if (original_center_x, original_center_y) != (bbox_center_x, bbox_center_y):
                ax.plot(original_center_x, original_center_y, 'go', markersize=3)
            
            # Find the minimum distance for this danger object
            for dist_info in results['min_distances']:
                if dist_info['danger_object_id'] == obj['id']:
                    distance = dist_info['distance']
                    closest_line_id = dist_info['closest_line']
                    
                    # Display distance information
                    if is_same_category_as_calibration and calibration_distance is not None:
                        # For calibration objects, only show calibration distance, not distance to power lines
                        ax.text(center_x + 5, center_y - 5, 
                               f"Calibration:{calibration_distance}m", 
                               color='black', fontsize=8, fontweight='bold')
                        # Add calibration object label
                        ax.text(center_x + 5, center_y + bbox[3] - bbox[1] + 15, 
                               "Reference", color='black', fontsize=8, fontweight='bold')
                    else:
                        # Calculate the closest point on the power line segment
                        segment_start = np.array(dist_info['closest_point_info']['segment_start'])
                        segment_end = np.array(dist_info['closest_point_info']['segment_end'])
                        danger_center = np.array([bbox_center_x, bbox_center_y])
                        
                        # Calculate the shortest distance from danger object to the line segment
                        t = max(0, min(1, np.dot(danger_center - segment_start, segment_end - segment_start) / np.dot(segment_end - segment_start, segment_end - segment_start)))
                        closest_point = segment_start + t * (segment_end - segment_start)
                        
                        # Draw the shortest distance line from danger object to closest point on power line
                        # 根据draw_distance_from参数选择绘制起点
                        if draw_distance_from == 'center':
                            # 从危险物上方边框中点开始绘制
                            start_x = (bbox[0] + bbox[2]) / 2  # 上方边框的水平中点
                            start_y = bbox[1]  # 上方边框的y坐标
                        elif 'selected_point' in obj:
                            # 从选择的点开始绘制
                            start_x, start_y = obj['selected_point']
                        else:
                            # 默认从危险物中心开始绘制，使用边界框的几何中心
                            start_x, start_y = bbox_center_x, bbox_center_y
                        ax.plot([start_x, closest_point[0]], [start_y, closest_point[1]], 'b--', linewidth=2, alpha=0.8, label='Min Distance' if obj['id'] == '0' else "")
                        
                        # For normal hazardous objects, display calculated distance ON the shortest distance line
                        # Calculate midpoint of the line segment
                        midpoint_x = (start_x + closest_point[0]) / 2
                        midpoint_y = (start_y + closest_point[1]) / 2
                        # Display distance text at midpoint
                        ax.text(midpoint_x, midpoint_y, f"{distance:.1f}m", 
                               color='white', backgroundcolor='blue', fontsize=10, fontweight='bold',
                               ha='center', va='center')
                    break
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', label='Calibration Object'),
            # Patch(facecolor='yellow', edgecolor='yellow', label='Similar Object'),
            Patch(facecolor='red', edgecolor='red', label='Danger Object'),
            # Patch(facecolor='green', edgecolor='green', label='Selected Closest Point'),
            # Patch(facecolor='none', edgecolor='green', linestyle='--', label='Center to Closest Point'),
            Patch(facecolor='none', edgecolor='green', label='Power Line'),
            Patch(facecolor='none', edgecolor='blue', label='Reference Tower'),
            Patch(facecolor='none', edgecolor='blue', linestyle='--', label='Min Distance')
        ]
        # if show_all_points_in_3d:
        #     legend_elements.append(Patch(facecolor='gray', edgecolor='gray', label=f'All Points ({max_all_points})'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加标定距离和相关系数信息
        calibration_info = []
        if calibration_class_id is not None:
            calibration_info.append(f"Calibration ID: {calibration_class_id}")
        if calibration_distance is not None:
            calibration_info.append(f"Calibration distance: {calibration_distance}m")
        
        # 检查是否有相关系数信息（如果有的话）
        if hasattr(analyzer, 'calibration_correlation') and analyzer.calibration_correlation is not None:
            calibration_info.append(f"相关系数: {analyzer.calibration_correlation:.4f}")
        
        if calibration_info or scale_factor is not None:
            # Display calibration information in the top-left corner
            calibration_text = "Calibration Info:\n"
            if calibration_info:
                calibration_text += "\n".join(calibration_info)
            if scale_factor is not None:
                if calibration_info:
                    calibration_text += "\n"
                calibration_text += f"Scale factor: {scale_factor:.2f}"
            ax.text(10, 30, calibration_text, 
                   fontsize=10, color='white', backgroundcolor='black', 
                   verticalalignment='top', bbox=dict(facecolor='black', alpha=0.7))
        
        # Add camera intrinsics information if available
        if hasattr(analyzer, 'fx') and analyzer.fx is not None:
            camera_info = "Camera Intrinsics:\n"
            camera_info += f"fx: {analyzer.fx:.2f}, fy: {analyzer.fy:.2f}\n"
            camera_info += f"cx: {analyzer.cx:.2f}, cy: {analyzer.cy:.2f}"
            
            # Display camera intrinsics in the bottom-left corner
            ax.text(10, image.shape[0] - 10, camera_info, 
                   fontsize=10, color='white', backgroundcolor='black', 
                   verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.7))
        
        # Set title
        ax.set_title('User Data - Distance Analysis (with Min Distance Lines)')
        ax.axis('off')
        
        # Save visualization result
        output_path = os.path.join(output_dir, "user_visualization.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization result saved to: {output_path}")
        
        # Add 3D point cloud visualization functionality
        try:
            # Call SpatialAnalyzer's 3D visualization methods
            if hasattr(analyzer, 'visualize_3d_point_cloud'):
                # Save 3D point cloud visualization with default filename
                cloud_output_path = os.path.join(output_dir, "user_3d_visualization.png")
                # Correctly call 3D point cloud visualization method
                analyzer.visualize_3d_point_cloud(output_path=cloud_output_path, show_plot=False, show_all_points=show_all_points_in_3d, max_all_points=max_all_points)
                if show_all_points_in_3d:
                    print(f"Note: All point cloud display enabled, showing up to {max_all_points} points")
                print(f"3D point cloud visualization saved to: {cloud_output_path}")
                
                # Try to generate combined visualization
                if hasattr(analyzer, 'create_combined_visualization'):
                    combined_output_path = os.path.join(output_dir, "user_combined_visualization.png")
                    # Correctly call combined visualization method
                    analyzer.create_combined_visualization(image_path, distance_results=results, 
                                                      output_path=combined_output_path, show_plot=False, 
                                                      show_all_points=show_all_points_in_3d, max_all_points=max_all_points)
                    print(f"Combined visualization saved to: {combined_output_path}")
        except Exception as cloud_e:
            print(f"Error during 3D visualization: {cloud_e}")
            import traceback
            traceback.print_exc()
        
        return output_path
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results_summary(results):
    """
    Print results summary
    
    Args:
        results: Distance calculation results
    """
    print("\nDistance calculation results summary:")
    print(f"  - Analysis time: {results['timestamp']}")
    print(f"  - Number of hazardous objects: {results['danger_objects_count']}")
    print(f"  - Number of power lines: {results['power_lines_count']}")
    print(f"  - Scale factor used: {results['scale_factor_used']}")
    print("\nDetailed distance information:")
    
    for dist_info in results['min_distances']:
        obj_id = dist_info['danger_object_id']
        class_id = dist_info['class_id']
        distance = dist_info['distance']
        closest_line = dist_info['closest_line']
        
        print(f"  Hazardous object #{obj_id} (class {class_id})")
        print(f"    - Minimum distance: {distance:.2f} meters")
        print(f"    - Closest power line: {closest_line}")
        
        # Check for dangerous proximity
        if distance < 5.0:  # Assuming 5 meters as safety threshold
            print(f"    ⚠️  Warning: Distance too close!")

def main():
    print("="*60)
    print("        用户数据处理 - 危险物与输电线距离分析        ")
    print("="*60)
    print("说明：您可以选择以下三种方式之一设置比例因子：")
    print("  1. 手动设置scale_factor值")
    print("  2. 使用标定物自动计算：设置calibration_class_id和calibration_distance")
    print("  3. 不设置以上参数，使用默认值")
    print("\n如果同时设置了手动比例因子和标定信息，将优先使用标定信息计算比例因子")
    print("\n您还可以提供regions_json_path参数来启用区域化深度平滑功能")
    print("\n您可以通过enable_smoothing参数控制是否启用输电线深度平滑功能")
    print("\n您可以通过enable_y_factor_adjustment参数控制是否启用基于图像垂直位置的深度调整（默认启用）")
    print("="*60)
    """
    主函数：演示如何使用用户自己的数据进行危险物与输电线距离计算
    """
    print("="*60)
    print("        用户数据处理 - 危险物与输电线距离分析        ")
    print("="*60)
    print("说明：您可以选择以下三种方式之一设置比例因子：")
    print("  1. 手动设置scale_factor值")
    print("  2. 使用标定物自动计算：设置calibration_class_id和calibration_distance")
    print("  3. 不设置以上参数，使用默认值")
    print("\n如果同时设置了手动比例因子和标定信息，将优先使用标定信息计算比例因子")
    print("\n您还可以提供regions_json_path参数来启用区域化深度平滑功能")
    print("="*60)
    
    # 这里是示例路径，用户需要根据自己的实际情况修改
    # 用户需要修改以下路径为自己的文件路径
    user_config = {
        'image_path': r"/Users/oldshen/Desktop/测距实验/6/原数据/6.jpg",          # 用户的图像文件
        'annotation_path': r"/Users/oldshen/Desktop/测距实验/6/原数据/6.txt", # YOLO格式的txt标注
        'segmentation_path': r"/Users/oldshen/Desktop/测距实验/6/原数据/6.json", # JSON格式的分割标注  
        'depth_map_path': r'/Users/oldshen/Desktop/测距实验/6/原数据/6_depth.npy',  # 如果有现成的深度图，可以提供路径
        'regions_json_path': None,  # 区域划分JSON文件路径（可选，用于区域化深度平滑）
        'enable_smoothing': False,   # 是否启用输电线深度平滑功能（可选，默认启用）
        'enable_y_factor_adjustment': False,  # 是否启用基于图像垂直位置的深度调整（可选，默认启用）
        'output_dir': r"/Users/oldshen/Desktop/测距实验/6/测距结果",  # 输出结果保存目录
        # 以下参数选择一种方式设置：
        'scale_factor': None,  # 手动设置比例因子（如果使用手动方式）
        'calibration_class_id': 1,  # 标定物的类别ID（如果使用标定物方式）
        'calibration_distance':100, # 标定物到摄像头的实际距离（米）（如果使用标定物方式）
        'show_all_points_in_3d': True,  # 是否显示所有点云（可选，默认显示）
        'max_all_points': 1000,  # 最大显示点云数量（可选，默认10000）
        'distance_method': 'segment', # 距离计算方法：'segment'（点到线段，默认，高效）或 'point'（点到点，精确但较慢）
        'draw_distance_from': 'center', # 距离线段的绘制起点：'center'（危险物中心）或'selected'（选择的点）
        # 相机内参参数（可选）：以下两种方式选择一种
        # 方式1：直接提供3x3相机矩阵
        'camera_matrix': None,  # 示例: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # 方式2：单独提供相机参数
        'fx': None,  # 焦距x方向
        'fy': None,  # 焦距y方向
        'cx': None,  # 主点x坐标
        'cy': None   # 主点y坐标
    }
    
    print("\n请确保以下路径正确：")
    for key, value in user_config.items():
        print(f"  - {key}: {value}")
    print(f"\n输电线深度平滑功能: {'已启用' if user_config.get('enable_smoothing', True) else '已禁用'}")
    print(f"区域化深度平滑功能: {'已启用' if user_config.get('regions_json_path') else '未启用'}")
    distance_method = user_config.get('distance_method', 'segment')
    method_desc = '点到线段距离（高效）' if distance_method == 'segment' else '点到点距离（精确但较慢）'
    print(f"距离计算方法: {distance_method} ({method_desc})")
    print("\n注意：请修改上述路径为您自己的实际文件路径")
    print("      如果使用标定物方式，请确保设置正确的calibration_class_id和calibration_distance")
    
    try:
        # 1. 准备用户数据
        print("\n[1/4] 准备用户数据...")
        depth_map, danger_objects, segmentation_path, scale_factor, camera_params = prepare_user_data(
            user_config['image_path'],
            user_config['annotation_path'], 
            user_config['segmentation_path'],
            user_config['depth_map_path'],
            user_config['scale_factor'],
            user_config.get('calibration_class_id'),  # 可选的标定物类别ID
            user_config.get('calibration_distance'),  # 可选的标定物实际距离
            user_config.get('camera_matrix'),         # 可选的相机内参矩阵
            user_config.get('fx'),                    # 可选的焦距x方向
            user_config.get('fy'),                    # 可选的焦距y方向
            user_config.get('cx'),                    # 可选的主点x坐标
            user_config.get('cy'),                    # 可选的主点y坐标
            user_config.get('regions_json_path')      # 可选的区域划分JSON文件路径
        )
        
        # 2. 计算距离
        print("\n[2/4] 计算危险物与输电线之间的距离...")
        analyzer, results = calculate_distances(
            depth_map, 
            danger_objects,
            segmentation_path,
            scale_factor,
            regions_json_path=user_config.get('regions_json_path'),
            enable_smoothing=user_config.get('enable_smoothing', True),
            enable_y_factor_adjustment=user_config.get('enable_y_factor_adjustment', True),
            calibration_class_id=user_config.get('calibration_class_id'),
            calibration_distance=user_config.get('calibration_distance'),
            distance_method=user_config.get('distance_method', 'segment'),
            **camera_params  # 解包相机内参参数
        )
        
        # 3. 打印结果摘要
        print("\n[3/4] 显示结果摘要...")
        print_results_summary(results)
        
        # 4. 保存结果和可视化
        print("\n[4/4] 保存结果和可视化...")
        output_dir = user_config.get('output_dir', 'user_results')  # 使用配置的输出目录，默认'user_results'
        save_results(results, output_dir=output_dir)
        visualize_with_user_data(
            user_config['image_path'], 
            analyzer, 
            results,
            output_dir=output_dir,
            calibration_class_id=user_config.get('calibration_class_id'),
            calibration_distance=user_config.get('calibration_distance'),
            scale_factor=scale_factor,
            show_all_points_in_3d=user_config.get('show_all_points_in_3d', False),
            max_all_points=user_config.get('max_all_points', 10000),
            draw_distance_from=user_config.get('draw_distance_from', 'center')
        )
        
        print("\n" + "="*60)
        print(f"处理完成！请在 {output_dir} 目录查看结果")
        print(f"- 2D距离可视化: {os.path.join(output_dir, 'user_visualization.png')}")
        print(f"- 三维点云可视化: {os.path.join(output_dir, 'user_3d_visualization.png')} (如果生成成功)")
        print(f"- 组合可视化图: {os.path.join(output_dir, 'user_combined_visualization.png')} (如果生成成功)")
        print(f"- 距离计算结果: {os.path.join(output_dir, 'distance_results.json')}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n错误：找不到文件 - {e}")
        print("请确保所有文件路径正确")
    except Exception as e:
        print(f"\n错误：处理过程中出现问题 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



