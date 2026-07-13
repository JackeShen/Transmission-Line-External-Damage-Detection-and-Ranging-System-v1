import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Union
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

class SpatialAnalyzer:
    """
    三维空间分析器，用于构建图像的三维空间并计算物体之间的距离
    """
    
    def __init__(self):
        """
        初始化空间分析器
        """
        self.depth_map = None
        self.image_width = 0
        self.image_height = 0
        self.danger_objects = []  # 危险物列表
        self.power_lines = []     # 输电线列表
        self.scale_factor = None  # 深度标定比例因子
        # 相机内参
        self.camera_matrix = None  # 3x3相机内参矩阵
        self.fx = None  # x方向焦距
        self.fy = None  # y方向焦距
        self.cx = None  # 主点x坐标
        self.cy = None  # 主点y坐标
        # 标定物信息
        self.calibration_object = None  # 标定物对象
        self.calibration_depth = None   # 标定物深度值（最大允许深度）
        
    def set_depth_map(self, depth_map: np.ndarray):
        """
        设置深度图
        
        Args:
            depth_map: 深度图numpy数组
        """
        self.depth_map = depth_map
        self.image_height, self.image_width = depth_map.shape
        
    def set_scale_factor(self, scale_factor: float):
        """
        设置深度标定比例因子
        
        Args:
            scale_factor: 从YOLODepthExtractor获取的比例因子
        """
        self.scale_factor = scale_factor
    
    def set_calibration_object(self, calibration_object: Dict):
        """
        设置标定物信息
        
        Args:
            calibration_object: 标定物对象，包含bbox和depth信息
        """
        self.calibration_object = calibration_object
        if 'depth' in calibration_object:
            self.calibration_depth = calibration_object['depth']
            print(f"已设置标定物深度约束: {self.calibration_depth}")
    
    def set_image_size(self, width: int, height: int):
        """
        设置图像尺寸
        
        Args:
            width: 图像宽度
            height: 图像高度
        """
        self.image_width = width
        self.image_height = height
    
    def smooth_power_line_depths(self, apply_global_smooth: bool = True, global_sigma: float = 1.0, 
                               line_direction_smooth: bool = True, depth_gradient_factor: float = 0.1,
                               regions_json_path: str = None, enable_y_factor_adjustment: bool = True):
        """
        平滑输电线的深度值，使其沿着线路方向呈现自然的渐变
        
        Args:
            apply_global_smooth: 是否应用全局高斯平滑到深度图
            global_sigma: 全局平滑的高斯核标准差
            line_direction_smooth: 是否应用线路方向的深度渐变
            depth_gradient_factor: 深度渐变因子，控制深度变化的强度
            regions_json_path: 区域划分JSON文件路径，如果提供则按区域进行平滑
            enable_y_factor_adjustment: 是否启用基于图像垂直位置的深度调整，使图像上方的点深度增加
        
        Returns:
            bool: 是否成功平滑
        """
        if self.depth_map is None or not self.power_lines:
            print("错误：深度图或输电线数据为空")
            return False
        
        # 创建深度图的副本进行处理
        processed_depth_map = self.depth_map.copy()
        
        # 标记是否使用区域化平滑
        using_regional_smooth = regions_json_path is not None
        
        # 如果提供了区域文件，加载区域信息
        regions = []
        if using_regional_smooth:
            regions = self.load_power_line_regions(regions_json_path)
            if not regions:
                print("警告：无法加载区域文件或区域为空，将使用全局平滑模式")
                using_regional_smooth = False
        
        # 应用全局高斯平滑（如果启用且未使用区域化平滑）
        if apply_global_smooth and not using_regional_smooth:
            processed_depth_map = gaussian_filter(processed_depth_map, sigma=global_sigma)
            print(f"应用全局高斯平滑，sigma={global_sigma}")
        
        # 如果使用区域化平滑，为每个区域创建掩码
        region_masks = []
        if using_regional_smooth:
            for region in regions:
                mask = self.create_region_mask(region)
                if mask is not None:
                    region_masks.append((region, mask))
            print(f"创建了 {len(region_masks)} 个区域掩码")
        
        # 应用深度平滑和渐变调整
        if line_direction_smooth or using_regional_smooth:
            # 首先收集所有输电线的点位置和深度信息
            all_line_positions = []
            all_line_depths = []
            
            for line in self.power_lines:
                if 'points' not in line:
                    continue
                
                for (x, y) in line['points']:
                    try:
                        x_int, y_int = int(round(float(x))), int(round(float(y)))
                        if 0 <= x_int < self.image_width and 0 <= y_int < self.image_height:
                            # 收集线路点及其周围点
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    nx, ny = x_int + dx, y_int + dy
                                    if 0 <= nx < self.image_width and 0 <= ny < self.image_height:
                                        all_line_positions.append((nx, ny))
                                        all_line_depths.append(processed_depth_map[ny, nx])
                    except (ValueError, TypeError):
                        continue
            
            # 计算全局统计信息（用于默认值），只保留深度值较小的部分
            global_depth_threshold = np.percentile(all_line_depths, 25) if all_line_depths else 0
            filtered_global_depths = [d for d in all_line_depths if d < global_depth_threshold]
            
            # 如果筛选后没有足够的数据，使用原始数据
            if len(filtered_global_depths) < len(all_line_depths) * 0.1:
                filtered_global_depths = all_line_depths
            
            print(f"  全局深度筛选: 原始点数={len(all_line_depths)}, 筛选后点数={len(filtered_global_depths)}")
            print(f"  全局深度阈值={global_depth_threshold:.2f}, 筛选前平均深度={np.mean(all_line_depths):.2f}, 筛选后平均深度={np.mean(filtered_global_depths):.2f}")
            
            global_mean_depth = np.mean(filtered_global_depths) if filtered_global_depths else 0
            global_std_depth = np.std(filtered_global_depths) if filtered_global_depths else 1
            
            # 如果使用区域化平滑，按区域处理
            if using_regional_smooth:
                # 创建一个权重图，用于区域边界的平滑过渡
                weights = np.zeros_like(processed_depth_map, dtype=np.float32)
                
                # 为每个区域应用平滑
                for region_idx, (region, mask) in enumerate(region_masks):
                    # 获取区域特定参数
                    params = region['smooth_params']
                    region_sigma = params.get('sigma', 1.0)
                    region_gradient_factor = params.get('gradient_factor', 0.3)
                    apply_smooth = params.get('apply_smooth', True)
                    
                    print(f"处理区域 {region['id']}，类型: {region['type']}，参数: sigma={region_sigma}, gradient_factor={region_gradient_factor}")
                    
                    # 在区域内应用高斯平滑（如果启用）
                    if apply_smooth:
                        # 创建区域的高斯掩码（边界平滑）
                        
                        # 创建一个临时深度图，只在区域内应用平滑
                        region_depth = processed_depth_map.copy()
                        
                        # 提取区域内的深度值
                        region_points = np.where(mask > 0)
                        if len(region_points[0]) > 0:
                            # 对区域进行高斯平滑，但只影响区域内的点
                            # 创建一个小区域进行局部平滑
                            smoothed_region = gaussian_filter(processed_depth_map, sigma=region_sigma)
                            
                            # 计算区域边界权重（使过渡更平滑）
                            boundary_mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
                            boundary_mask = np.clip(boundary_mask, 0, 1)
                            
                            # 混合原始深度和平滑深度
                            processed_depth_map = (smoothed_region * boundary_mask + 
                                                 processed_depth_map * (1 - boundary_mask))
                            
                            # 记录权重
                            weights = np.maximum(weights, boundary_mask)
                
                # 处理区域外的输电线部分（使用全局参数）
                # 计算未被任何区域覆盖的输电线点
                unweighted_mask = (weights < 0.1).astype(np.float32)
                
                if np.sum(unweighted_mask) > 0 and line_direction_smooth:
                    self._apply_directional_smooth(processed_depth_map, all_line_positions, 
                                                global_mean_depth, global_std_depth, 
                                                depth_gradient_factor, mask=unweighted_mask)
            else:
                # 传统的全局平滑模式
                for line_idx, line in enumerate(self.power_lines):
                    if 'points' not in line or len(line['points']) < 2:
                        continue
                    
                    # 获取输电线的所有点
                    points = line['points']
                    
                    # 计算输电线的总长度（像素空间）
                    total_length = 0
                    for i in range(len(points) - 1):
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]
                        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        total_length += segment_length
                    
                    if total_length == 0:
                        continue
                    
                    # 为输电线创建一个掩码
                    line_mask = np.zeros_like(self.depth_map, dtype=bool)
                    line_depth_values = []
                    line_positions = []
                    
                    # 填充掩码并收集深度值
                    for (x, y) in points:
                        try:
                            x_int, y_int = int(round(float(x))), int(round(float(y)))
                            if 0 <= x_int < self.image_width and 0 <= y_int < self.image_height:
                                line_mask[y_int, x_int] = True
                                # 扩展掩码以包含线周围的像素
                                for dy in range(-1, 2):
                                    for dx in range(-1, 2):
                                        nx, ny = x_int + dx, y_int + dy
                                        if 0 <= nx < self.image_width and 0 <= ny < self.image_height:
                                            line_mask[ny, nx] = True
                                            line_depth_values.append(processed_depth_map[ny, nx])
                                            line_positions.append((nx, ny))
                        except (ValueError, TypeError):
                            continue
                    
                    if not line_depth_values:
                        continue
                    
                    # 计算线路上的深度统计信息，只保留深度值较小的部分（解决与背景天空混淆问题）
                    # 计算深度值的25百分位数，只保留小于该百分位数的深度值（距离较近的点）
                    depth_threshold = np.percentile(line_depth_values, 25)
                    filtered_depth_values = [d for d in line_depth_values if d < depth_threshold]
                    
                    # 如果筛选后没有足够的数据，使用原始数据
                    if len(filtered_depth_values) < len(line_depth_values) * 0.1:
                        filtered_depth_values = line_depth_values
                    
                    print(f"  输电线 {line_idx} 深度筛选: 原始点数={len(line_depth_values)}, 筛选后点数={len(filtered_depth_values)}")
                    print(f"  深度阈值={depth_threshold:.2f}, 筛选前平均深度={np.mean(line_depth_values):.2f}, 筛选后平均深度={np.mean(filtered_depth_values):.2f}")
                    
                    # 使用筛选后的数据计算统计信息
                    mean_depth = np.mean(filtered_depth_values)
                    std_depth = np.std(filtered_depth_values)
                    
                    # 确定线路的起始和结束点
                    start_point = points[0]
                    end_point = points[-1]
                    
                    # 计算线路的实际方向向量
                    direction_vec = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
                    
                    # 计算每个线段的累计距离，用于后续的相对位置计算
                    cumulative_distances = [0.0]
                    for i in range(len(points) - 1):
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]
                        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)** 2)
                        cumulative_distances.append(cumulative_distances[-1] + segment_length)
                    
                    # 更精确地计算点在线路上的位置
                    # 首先，为每个像素点找到最近的线路线段和在线段上的投影位置
                    optimized_line_depths = {}
                    
                    for (nx, ny) in line_positions:
                        min_dist = float('inf')
                        closest_segment_idx = 0
                        projection_ratio = 0.0
                        
                        # 找到距离该像素最近的线路线段
                        for i in range(len(points) - 1):
                            x1, y1 = points[i]
                            x2, y2 = points[i + 1]
                            
                            # 计算点到线段的距离和投影比例
                            segment_vec = np.array([x2 - x1, y2 - y1])
                            point_vec = np.array([nx - x1, ny - y1])
                            
                            # 线段长度的平方
                            segment_len_sq = np.dot(segment_vec, segment_vec)
                            
                            if segment_len_sq == 0:
                                continue
                            
                            # 计算投影比例 t
                            t = max(0, min(1, np.dot(point_vec, segment_vec) / segment_len_sq))
                            
                            # 计算投影点
                            projection_x = x1 + t * segment_vec[0]
                            projection_y = y1 + t * segment_vec[1]
                            
                            # 计算距离
                            dist = np.sqrt((nx - projection_x)**2 + (ny - projection_y)** 2)
                            
                            if dist < min_dist:
                                min_dist = dist
                                closest_segment_idx = i
                                projection_ratio = t
                        
                        # 计算该点在线路上的累计距离
                        segment_start_dist = cumulative_distances[closest_segment_idx]
                        segment_length = cumulative_distances[closest_segment_idx + 1] - segment_start_dist
                        point_cumulative_dist = segment_start_dist + projection_ratio * segment_length
                        
                        # 计算相对位置（0-1）
                        relative_position = point_cumulative_dist / total_length if total_length > 0 else 0.0
                        
                        # 智能确定深度渐变方向
                        # 考虑线路方向和相机视角
                        # 1. 检查线路是否从左上到右下或从右上到左下
                        diagonal_direction = abs(direction_vec[0]) > abs(direction_vec[1])
                        
                        # 2. 根据相机视角调整深度渐变（如果启用）
                        # 假设相机在图像下方，上方的物体应该更远
                        if enable_y_factor_adjustment:
                            y_factor = 1.0 - (ny / self.image_height) * 0.2  # 图像上方的点稍微增加深度
                        else:
                            y_factor = 1.0  # 不应用y_factor调整
                        
                        # 3. 组合多种因素计算深度因子
                        base_depth_factor = 0.95 + 0.15 * relative_position  # 基础渐变
                        adjusted_depth_factor = base_depth_factor * y_factor
                        
                        # 记录优化后的深度信息
                        optimized_line_depths[(nx, ny)] = {
                            'relative_position': relative_position,
                            'depth_factor': adjusted_depth_factor
                        }
                    
                    # 应用优化的深度调整
                    for (nx, ny), info in optimized_line_depths.items():
                        original_depth = processed_depth_map[ny, nx]
                        adjusted_depth = original_depth * info['depth_factor']
                        
                        # 智能限制深度变化范围
                        # 根据点在线路上的位置动态调整限制范围
                        position_factor = info['relative_position']
                        min_allowed_depth = mean_depth - std_depth * 0.5
                        max_allowed_depth = mean_depth + std_depth * (1.0 + position_factor * 1.5)  # 越远允许的变化越大
                        
                        # 应用非线性深度调整，使远处的深度变化更自然
                        if position_factor > 0.5:
                            # 对于远处的点，使用平方因子增加深度变化
                            adjusted_depth = original_depth * (info['depth_factor'] + position_factor * 0.1)
                        
                        adjusted_depth = max(min_allowed_depth, min(max_allowed_depth, adjusted_depth))
                        processed_depth_map[ny, nx] = adjusted_depth
                    
                    print(f"处理输电线 {line_idx}，平均深度: {mean_depth:.2f}，调整了 {len(line_positions)} 个点")
        
        # 更新深度图
        self.depth_map = processed_depth_map
        print("输电线深度平滑完成")
        return True
    
    def _apply_directional_smooth(self, depth_map, line_positions, mean_depth, std_depth, 
                                gradient_factor, mask=None):
        """
        对指定区域内的输电线点应用方向平滑
        
        Args:
            depth_map: 深度图
            line_positions: 输电线点位置列表
            mean_depth: 平均深度
            std_depth: 深度标准差
            gradient_factor: 渐变因子
            mask: 可选的掩码，指定要处理的区域
        """
        if not line_positions:
            return
        
        # 计算深度渐变的全局方向
        # 统计图像上方和下方的平均深度
        top_region = depth_map[:depth_map.shape[0]//3, :]
        bottom_region = depth_map[2*depth_map.shape[0]//3:, :]
        
        top_mean = np.mean(top_region)
        bottom_mean = np.mean(bottom_region)
        
        # 确定深度递增方向
        depth_increases_upward = top_mean > bottom_mean
        
        # 对每个点应用方向平滑
        for (nx, ny) in line_positions:
            # 检查是否在掩码内（如果提供了掩码）
            if mask is not None and mask[ny, nx] < 0.5:
                continue
            
            # 根据图像位置计算深度因子
            # 上方的点应该更远（深度更大）
            y_normalized = ny / depth_map.shape[0]
            
            if depth_increases_upward:
                # 深度向上增加
                depth_factor = 1.0 + (1.0 - y_normalized) * gradient_factor
            else:
                # 深度向上减少（需要反转）
                depth_factor = 1.0 + y_normalized * gradient_factor
            
            # 应用深度调整
            original_depth = depth_map[ny, nx]
            adjusted_depth = original_depth * depth_factor
            
            # 限制深度变化范围
            min_allowed_depth = mean_depth - std_depth * 0.5
            max_allowed_depth = mean_depth + std_depth * 1.5
            
            adjusted_depth = max(min_allowed_depth, min(max_allowed_depth, adjusted_depth))
            depth_map[ny, nx] = adjusted_depth
    
    def set_camera_intrinsics(self, fx: float = None, fy: float = None, cx: float = None, cy: float = None, 
                            camera_matrix: np.ndarray = None):
        """
        设置相机内参
        
        Args:
            fx: x方向焦距（像素）
            fy: y方向焦距（像素）
            cx: 主点x坐标（像素）
            cy: 主点y坐标（像素）
            camera_matrix: 3x3相机内参矩阵，如果提供则忽略单独的参数
        """
        if camera_matrix is not None:
            # 从相机矩阵中提取参数
            if camera_matrix.shape == (3, 3):
                self.camera_matrix = camera_matrix
                self.fx = camera_matrix[0, 0]
                self.fy = camera_matrix[1, 1]
                self.cx = camera_matrix[0, 2]
                self.cy = camera_matrix[1, 2]
                print(f"使用提供的相机内参矩阵: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            else:
                print(f"警告: 相机内参矩阵形状不正确，应为(3,3)，但得到{camera_matrix.shape}")
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            # 使用单独提供的参数
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            print(f"使用提供的相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            print("警告: 未提供有效的相机内参，将使用默认值")
            self.camera_matrix = None
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None
        
    def add_danger_objects(self, annotation_depths: Dict):
        """
        添加危险物（从YOLO标注中提取）
        
        Args:
            annotation_depths: 从YOLODepthExtractor获取的标注深度信息
        """
        self.danger_objects = []
        
        for idx, depth_info in annotation_depths.items():
            obj = {
                'id': idx,
                'class_id': depth_info['class_id'],
                'bbox': depth_info['bbox'],  # [x_min, y_min, x_max, y_max]
                'distance': depth_info.get('estimated_distance', None),
                'center_2d': self._calculate_bbox_center(depth_info['bbox']),
                'center_3d': None  # 将在build_3d_space中计算
            }
            self.danger_objects.append(obj)
    
    def load_power_line_annotations(self, json_path: str):
        """
        从JSON文件加载输电线分割标注
        
        Args:
            json_path: JSON分割标注文件路径
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查不同的JSON格式
            if 'segments' in data:
                # 原始格式：包含segments键
                self.power_lines = data['segments']
                print(f"成功加载 {len(self.power_lines)} 条输电线标注 (格式1: segments)")
            elif 'shapes' in data:
                # 用户提供的格式：包含shapes数组
                shapes = data['shapes']
                # 转换为我们需要的格式
                self.power_lines = []
                for i, shape in enumerate(shapes):
                    if shape.get('label') == 'cable' or shape.get('label') == 'wire':
                        line = {
                            'id': f'line_{i}',
                            'points': shape.get('points', []),
                            'label': shape.get('label', 'cable'),
                            'shape_type': shape.get('shape_type', 'polygon')
                        }
                        self.power_lines.append(line)
                print(f"成功加载 {len(self.power_lines)} 条输电线标注 (格式2: shapes)")
            else:
                # 尝试直接将data作为shapes数组处理
                if isinstance(data, list):
                    # 转换为我们需要的格式
                    self.power_lines = []
                    for i, shape in enumerate(data):
                        if shape.get('label') == 'cable' or shape.get('label') == 'wire':
                            line = {
                                'id': f'line_{i}',
                                'points': shape.get('points', []),
                                'label': shape.get('label', 'cable'),
                                'shape_type': shape.get('shape_type', 'polygon')
                            }
                            self.power_lines.append(line)
                    print(f"成功加载 {len(self.power_lines)} 条输电线标注 (格式3: direct list)")
                else:
                    # 未知格式
                    print(f"警告: 未知的JSON格式，尝试使用空列表")
                    self.power_lines = []
            
            # 验证加载的输电线数据
            valid_lines = []
            for i, line in enumerate(self.power_lines):
                if 'points' in line and line['points']:
                    # 确保点格式正确
                    points = line['points']
                    if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in points):
                        valid_lines.append(line)
                    else:
                        print(f"警告: 输电线 {i} 的点格式不正确")
                else:
                    print(f"警告: 输电线 {i} 没有有效的点数据")
            
            self.power_lines = valid_lines
            print(f"最终有效输电线数量: {len(self.power_lines)}")
            return len(self.power_lines) > 0
        except Exception as e:
            print(f"加载输电线标注出错: {e}")
            import traceback
            traceback.print_exc()
            self.power_lines = []
            return False
    
    def _calculate_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """
        计算边界框中心点
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            
        Returns:
            (center_x, center_y) 中心点坐标
        """
        x_min, y_min, x_max, y_max = bbox
        return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    
    def _get_depth_at_point(self, x: float, y: float) -> float:
        """
        获取指定像素点的深度值，并进行过滤处理
        
        Args:
            x: 像素x坐标（可能是浮点数）
            y: 像素y坐标（可能是浮点数）
            
        Returns:
            深度值（经过过滤处理）
        """
        if self.depth_map is None:
            return None
        
        try:
            # 将坐标转换为整数
            x_int = int(round(float(x)))
            y_int = int(round(float(y)))
            
            # 确保坐标在有效范围内
            x_int = max(0, min(self.image_width - 1, x_int))
            y_int = max(0, min(self.image_height - 1, y_int))
            
            depth = float(self.depth_map[y_int, x_int])
            
            # 过滤异常低的深度值
            # 计算深度图的统计信息
            valid_depths = self.depth_map[self.depth_map > 0]
            if len(valid_depths) > 0:
                # 使用深度图的10%分位数作为下限
                depth_lower_limit = np.percentile(valid_depths, 10)
                if depth < depth_lower_limit:
                    print(f"警告: 深度值 {depth:.4f} 低于下限 {depth_lower_limit:.4f}，将使用下限值")
                    depth = depth_lower_limit
            
            return depth
        except (ValueError, TypeError, IndexError) as e:
            print(f"获取深度值时出错: x={x}, y={y}, 错误={e}")
            return None
    
    def _convert_2d_to_3d(self, x: int, y: int, depth: float = None) -> Tuple[float, float, float]:
        """
        将2D像素坐标转换为3D空间坐标
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            depth: 深度值，如果为None则从深度图获取
            
        Returns:
            (x_3d, y_3d, z_3d) 三维坐标
        """
        if depth is None:
            depth = self._get_depth_at_point(x, y)
            if depth is None:
                return None
        
        # 智能深度约束 - 输电线深度不能小于标定物深度（符合物理规律：输电线架设在输电塔上）
        # 保持输电线的相对深度变化，确保自然的视觉效果
        original_depth = depth
        if self.calibration_depth is not None and depth < self.calibration_depth:
            # 对于小于标定物深度的点，使用相对深度缩放策略
            # 保持点之间的相对深度比例，同时确保不小于标定物深度
            # 计算深度缩放因子
            depth = self.calibration_depth
            print(f"点({x},{y})深度 {original_depth} 小于标定物深度 {self.calibration_depth}，已调整为 {depth}")
        
        # 计算实际距离（如果有比例因子）
        # 新逻辑：深度值越高，物体越远（成正比关系）
        # 将深度值转换为正数，符合常规三维空间表示（z轴正方向表示向前/远离相机）
        if self.scale_factor is not None and depth > 0:
            z_3d = depth * self.scale_factor 
        else:
            z_3d = depth  # 使用相对深度
        
        # 确保z_3d为正数（深度值始终应为正数）
        z_3d = abs(z_3d)
        
        # 检查是否提供了相机内参
        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
            # 使用用户提供的相机内参进行3D坐标转换
            # 图像坐标系: x向右, y向下
            # 3D坐标系: x向右, y向上, z向前(深度)
            # 使用针孔相机模型进行精确计算
            x_3d = ((x - self.cx) / self.fx) * z_3d
            y_3d = -((y - self.cy) / self.fy) * z_3d  # 翻转Y轴方向，使向上为正
            print(f"使用用户提供的相机内参进行3D转换: x={x}, y={y}, z={z_3d} -> ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})")
        else:
            # 使用原来的简化版本（假设相机参数或使用近似值）
            # 图像坐标系: x向右, y向下
            # 3D坐标系: x向右, y向上, z向前(深度)
            
            # 假设焦距（可以根据实际相机参数调整）
            focal_length = self.image_width * 0.8  # 近似值
            
            # 计算相对x和y，注意y轴需要翻转方向
            x_rel = (x - self.image_width / 2) / focal_length
            y_rel = -(y - self.image_height / 2) / focal_length  # 翻转Y轴方向，使向上为正
            
            # 计算3D坐标：x和y是横向和纵向偏移，z是深度
            x_3d = x_rel * z_3d
            y_3d = y_rel * z_3d
        
        return (x_3d, y_3d, z_3d)
    
    def _sample_points_in_bbox(self, bbox: List[int], num_samples: int = 9) -> List[Tuple[int, int]]:
        """
        从边界框内均匀采样多个点
        
        Args:
            bbox: 边界框坐标 [x_min, y_min, x_max, y_max]
            num_samples: 采样点数量，默认9个（3x3网格）
            
        Returns:
            采样点坐标列表 [(x1, y1), (x2, y2), ...]
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算网格大小
        grid_size = int(np.sqrt(num_samples))
        if grid_size * grid_size != num_samples:
            grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        # 生成网格点
        samples = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算采样点坐标（中心偏移）
                x = int(x_min + width * (i + 0.5) / grid_size)
                y = int(y_min + height * (j + 0.5) / grid_size)
                # 确保坐标在图像范围内
                x = max(0, min(self.image_width - 1, x))
                y = max(0, min(self.image_height - 1, y))
                samples.append((x, y))
                
                # 如果达到指定数量，提前返回
                if len(samples) == num_samples:
                    return samples
        
        return samples
    
    def build_3d_space(self, use_closest_point=True):
        """
        构建三维空间，计算所有物体的3D坐标
        
        Args:
            use_closest_point: 是否使用危险物边界框内离摄像头最近的点作为中心点
            
        Returns:
            bool: 是否构建成功
        """
        if self.depth_map is None:
            print("错误：请先设置深度图")
            return False
        
        # 计算危险物的3D坐标（包括中心点和采样点）
        for obj in self.danger_objects:
            x, y = obj['center_2d']
            
            if use_closest_point:
                # 找到边界框内深度值最小的点（离摄像头最近）
                x_min, y_min, x_max, y_max = obj['bbox']
                
                # 确保边界框在图像范围内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(self.image_width - 1, x_max)
                y_max = min(self.image_height - 1, y_max)
                
                # 获取边界框内的所有深度值
                region_depth = self.depth_map[y_min:y_max+1, x_min:x_max+1]
                valid_depths = region_depth[region_depth > 0]
                
                if len(valid_depths) > 0:
                    # 找到最小深度值的位置（深度值越小表示距离越近）
                    min_depth = valid_depths.min()
                    min_depth_indices = np.where(region_depth == min_depth)
                    
                    # 取第一个出现的最小深度值的位置
                    closest_y = y_min + min_depth_indices[0][0]
                    closest_x = x_min + min_depth_indices[1][0]
                    
                    # 更新危险物的中心点为最近点
                    obj['center_2d'] = (closest_x, closest_y)
                    obj['closest_point_2d'] = (closest_x, closest_y)
                    obj['closest_point_depth'] = min_depth
                    
                    print(f"危险物 {obj['id']} 中心点已更新为最近点: ({closest_x}, {closest_y}), 深度值: {min_depth:.4f}")
                else:
                    print(f"警告: 危险物 {obj['id']} 边界框内没有有效深度值")
            
            # 获取中心点的深度值
            x, y = obj['center_2d']
            depth = self._get_depth_at_point(x, y)
            
            if depth is not None:
                obj['center_3d'] = self._convert_2d_to_3d(x, y, depth)
                
                # 如果使用了最近点，同时记录最近点的3D坐标
                if use_closest_point and 'closest_point_depth' in obj:
                    obj['closest_point_3d'] = obj['center_3d']
            
            # 采样多个点
            samples = self._sample_points_in_bbox(obj['bbox'])
            obj['sampled_points_2d'] = samples
            obj['sampled_points_3d'] = []
            
            # 计算每个采样点的3D坐标
            for (sx, sy) in samples:
                sample_depth = self._get_depth_at_point(sx, sy)
                if sample_depth is not None:
                    sample_3d = self._convert_2d_to_3d(sx, sy, sample_depth)
                    if sample_3d:
                        obj['sampled_points_3d'].append({
                            '2d': (sx, sy),
                            '3d': sample_3d,
                            'depth': sample_depth
                        })
        
        # 计算输电线每个点的3D坐标
        self.power_lines_3d = []
        for line_idx, line in enumerate(self.power_lines):
            if 'points' in line:
                line['points_3d'] = []
                line_3d = []
                line_depths = []
                for (x, y) in line['points']:
                    depth = self._get_depth_at_point(x, y)
                    if depth is not None:
                        point_3d = self._convert_2d_to_3d(x, y, depth)
                        line['points_3d'].append({
                            '2d': (x, y),
                            '3d': point_3d,
                            'depth': depth
                        })
                        line_3d.append(point_3d)
                        line_depths.append(depth)
                
                # 输出每条输电线的深度分布信息
                if line_depths:
                    min_depth = min(line_depths)
                    max_depth = max(line_depths)
                    avg_depth = sum(line_depths) / len(line_depths)
                    print(f"输电线 {line_idx} 深度分布: 最小={min_depth:.2f}, 最大={max_depth:.2f}, 平均={avg_depth:.2f}, 点数={len(line_depths)}")
                    print(f"  深度范围: {min_depth:.2f} - {max_depth:.2f}, 深度变化率: {(max_depth-min_depth)/min_depth*100:.1f}%")
                
                # 对输电线进行悬链线拟合
                if line_3d:
                    fitted_line_3d = self._catenary_fit(line_3d)
                    # 更新3D点
                    for i, point_info in enumerate(line['points_3d']):
                        if i < len(fitted_line_3d):
                            point_info['3d'] = fitted_line_3d[i]
                    
                    self.power_lines_3d.append(fitted_line_3d)
                else:
                    self.power_lines_3d.append(line_3d)
        
        print("三维空间构建完成")
        return True
    
    def _catenary_fit(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        使用悬链线模型拟合输电线
        
        Args:
            points: 输电线的3D点列表
            
        Returns:
            拟合后的3D点列表
        """
        if len(points) < 2:
            return points
        
        try:
            from scipy.optimize import curve_fit
            import numpy as np
            
            # 提取x和z坐标（假设x是水平方向，z是垂直方向）
            x = np.array([p[0] for p in points])
            z = np.array([p[2] for p in points])
            
            # 悬链线方程：z = a * cosh((x - c)/a) + d
            def catenary(x, a, c, d):
                return a * np.cosh((x - c) / a) + d
            
            # 初始参数估计
            x_min, x_max = np.min(x), np.max(x)
            z_min, z_max = np.min(z), np.max(z)
            initial_guess = [1.0, (x_min + x_max) / 2, z_min]
            
            # 进行拟合
            popt, _ = curve_fit(catenary, x, z, p0=initial_guess, maxfev=10000)
            
            # 使用拟合参数重新计算z坐标
            fitted_z = catenary(x, *popt)
            
            # 重建拟合后的3D点
            fitted_points = []
            for i, (x_val, y_val, z_val) in enumerate(points):
                # 保持x和y坐标不变，只更新z坐标
                fitted_points.append((x_val, y_val, fitted_z[i]))
            
            print(f"悬链线拟合完成，参数: a={popt[0]:.2f}, c={popt[1]:.2f}, d={popt[2]:.2f}")
            return fitted_points
        except ImportError:
            print("警告：scipy未安装，无法进行悬链线拟合")
            return points
        except Exception as e:
            print(f"悬链线拟合出错: {e}")
            return points
    
    def _euclidean_distance(self, point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            point1: 第一个点的三维坐标
            point2: 第二个点的三维坐标
            
        Returns:
            两点之间的距离
        """
        if point1 is None or point2 is None:
            return float('inf')
        
        return np.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2 +
            (point1[2] - point2[2]) ** 2
        )
    
    def _point_to_line_segment_distance(self, point: Tuple[float, float, float], 
                                       line_start: Tuple[float, float, float], 
                                       line_end: Tuple[float, float, float]) -> float:
        """
        计算点到线段的最短距离
        
        Args:
            point: 三维点坐标
            line_start: 线段起点
            line_end: 线段终点
            
        Returns:
            点到线段的最短距离
        """
        # 线段向量
        line_vec = np.array([
            line_end[0] - line_start[0],
            line_end[1] - line_start[1],
            line_end[2] - line_start[2]
        ])
        
        # 点到线段起点的向量
        point_vec = np.array([
            point[0] - line_start[0],
            point[1] - line_start[1],
            point[2] - line_start[2]
        ])
        
        # 线段长度的平方
        line_len_sq = np.dot(line_vec, line_vec)
        
        # 计算投影比例
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        
        # 计算投影点
        projection = np.array([
            line_start[0] + t * line_vec[0],
            line_start[1] + t * line_vec[1],
            line_start[2] + t * line_vec[2]
        ])
        
        # 计算点到投影点的距离
        return self._euclidean_distance(tuple(projection), point)
    
    def calculate_min_distance_to_power_lines(self, distance_method: str = 'segment'):
        """
        计算每个危险物到输电线的最小距离
        
        Args:
            distance_method: 距离计算方法
                'segment': 点到线段距离（默认，高效准确）
                'point': 点到点距离（低效，遍历所有线段点）
                
        Returns:
            List[Dict]: 包含距离信息的列表
        """
        results = []
        
        # 收集输电线数据
        if not self.power_lines:
            print("警告：没有输电线数据")
            return results
        
        if distance_method == 'segment':
            # 点到线段距离方法：收集线段信息并构建kd-tree
            power_line_segments = []
            for line_idx, line in enumerate(self.power_lines):
                if 'points_3d' not in line or not line['points_3d']:
                    continue
                    
                # 遍历输电线的所有线段
                for i in range(len(line['points_3d']) - 1):
                    point1_info = line['points_3d'][i]
                    point2_info = line['points_3d'][i + 1]
                    
                    # 跳过没有3D坐标的点
                    if point1_info['3d'] is None or point2_info['3d'] is None:
                        continue
                    
                    # 保存线段信息
                    power_line_segments.append({
                        'line_id': line.get('id', line_idx),
                        'start_point': point1_info['3d'],
                        'end_point': point2_info['3d'],
                        'start_2d': point1_info['2d'],
                        'end_2d': point2_info['2d']
                    })
            
            if not power_line_segments:
                print("警告：没有有效的输电线线段")
                return results
            
            # 提取所有线段的中点用于kd-tree索引
            segment_midpoints = []
            for segment in power_line_segments:
                mid_x = (segment['start_point'][0] + segment['end_point'][0]) / 2
                mid_y = (segment['start_point'][1] + segment['end_point'][1]) / 2
                mid_z = (segment['start_point'][2] + segment['end_point'][2]) / 2
                segment_midpoints.append([mid_x, mid_y, mid_z])
            
            # 构建kd-tree
            kd_tree = KDTree(segment_midpoints)
            print("使用点到线段距离方法，共{}个线段".format(len(power_line_segments)))
        else:  # 'point' 方法
            # 点到点距离方法：收集所有输电线的3D点
            power_line_points = []
            for line_idx, line in enumerate(self.power_lines):
                if 'points_3d' not in line or not line['points_3d']:
                    continue
                    
                # 遍历输电线的所有点
                for point_info in line['points_3d']:
                    if point_info['3d'] is not None:
                        power_line_points.append({
                            'line_id': line.get('id', line_idx),
                            'point_3d': point_info['3d'],
                            'point_2d': point_info['2d']
                        })
            
            if not power_line_points:
                print("警告：没有有效的输电线点")
                return results
            
            print("使用点到点距离方法，共{}个输电线点".format(len(power_line_points)))
        
        for obj_idx, obj in enumerate(self.danger_objects):
            # 排除标定物
            if hasattr(self, 'calibration_object') and self.calibration_object:
                # 如果标定物具有class_id属性，根据类别ID排除
                if 'class_id' in self.calibration_object and 'class_id' in obj:
                    if obj['class_id'] == self.calibration_object['class_id']:
                        print(f"跳过标定物 {obj_idx} (ID: {obj['id']}, 类别: {obj['class_id']})")
                        continue
                # 否则根据ID排除（兼容旧的实现）
                else:
                    obj_id = str(obj['id'])
                    calibration_id = str(self.calibration_object.get('id', ''))
                    if obj_id == calibration_id:
                        print(f"跳过标定物 {obj_idx} (ID: {obj['id']})")
                        continue
                
            if not obj.get('sampled_points_3d') and obj['center_3d'] is None:
                print(f"警告：危险物 {obj_idx} 没有3D坐标")
                continue
            
            min_distance = float('inf')
            closest_line_id = None
            closest_point_info = None
            closest_sample = None
            
            # 获取所有要计算距离的点（优先使用最近点，然后是采样点和中心点）
            points_to_check = []
            
            # 如果有最近点，优先添加到列表前面
            if 'closest_point_3d' in obj and obj['closest_point_3d'] is not None:
                points_to_check.append((obj['closest_point_3d'], obj['closest_point_2d']))
                print(f"危险物 {obj['id']} 使用最近点进行距离计算: {obj['closest_point_2d']}")
            
            # 添加采样点
            if obj.get('sampled_points_3d'):
                for sample in obj['sampled_points_3d']:
                    if sample['3d']:
                        # 避免重复添加最近点
                        if 'closest_point_2d' not in obj or sample['2d'] != obj['closest_point_2d']:
                            points_to_check.append((sample['3d'], sample['2d']))
            
            # 添加中心点（如果不是最近点）
            if obj['center_3d'] and ('closest_point_2d' not in obj or obj['center_2d'] != obj['closest_point_2d']):
                points_to_check.append((obj['center_3d'], obj['center_2d']))
            
            # 如果没有可用的3D点
            if not points_to_check:
                print(f"警告：危险物 {obj_idx} 没有可用的3D点")
                continue
            
            if distance_method == 'segment':
                # 点到线段距离方法
                # 对于每个危险物的点，使用kd-tree查找最近的线段
                for point_3d, point_2d in points_to_check:
                    # 查询最近的k个线段（k=5，增加找到真实最近线段的概率）
                    k = min(5, len(segment_midpoints))
                    distances, indices = kd_tree.query(point_3d, k=k)
                    
                    # 确保indices是可迭代的
                    if not isinstance(indices, (list, np.ndarray)):
                        indices = [indices]
                    
                    # 计算到这些候选线段的实际距离
                    for idx in indices:
                        segment = power_line_segments[idx]
                        distance = self._point_to_line_segment_distance(
                            point_3d,
                            segment['start_point'],
                            segment['end_point']
                        )
                        
                        # 更新最小距离
                        if distance < min_distance:
                            min_distance = distance
                            closest_line_id = segment['line_id']
                            closest_point_info = {
                                'line_id': closest_line_id,
                                'segment_start': segment['start_2d'],
                                'segment_end': segment['end_2d']
                            }
                            closest_sample = {
                                'sample_2d': point_2d,
                                'sample_3d': point_3d
                            }
            else:  # 'point' 方法
                # 点到点距离方法
                # 对于每个危险物的点，计算到所有输电线点的距离
                for point_3d, point_2d in points_to_check:
                    for pl_point in power_line_points:
                        distance = self._euclidean_distance(point_3d, pl_point['point_3d'])
                        
                        # 更新最小距离
                        if distance < min_distance:
                            min_distance = distance
                            closest_line_id = pl_point['line_id']
                            closest_point_info = {
                                'line_id': closest_line_id,
                                'point_2d': pl_point['point_2d'],
                                'point_3d': pl_point['point_3d']
                            }
                            closest_sample = {
                                'sample_2d': point_2d,
                                'sample_3d': point_3d
                            }
            
            # 保存结果
            result = {
                'danger_object_id': obj['id'],
                'class_id': obj['class_id'],
                'distance': min_distance,
                'closest_line': closest_line_id,
                'closest_point_info': closest_point_info,
                'danger_center_2d': obj['center_2d'],
                'danger_center_3d': obj['center_3d'],
                'closest_sample': closest_sample,
                'sampled_points_count': len(obj.get('sampled_points_3d', []))
            }
            results.append(result)
            
            # 输出结果
            print(f"危险物 {obj['id']} (类别 {obj['class_id']}) 到输电线的最小距离: {min_distance:.2f} 米")
            if closest_line_id is not None:
                print(f"  最近的输电线ID: {closest_line_id}")
                if closest_sample:
                    print(f"  最近的采样点2D: {closest_sample['sample_2d']}, 3D: {closest_sample['sample_3d']}")
        
        return results
    
    def calculate_all_distances(self, distance_method: str = 'segment'):
        """
        计算所有危险物与所有输电线之间的距离
        
        Returns:
            Dict: 包含所有距离计算结果的字典
        """
        # 确保已构建三维空间
        if not self._is_3d_space_built():
            print("错误：请先调用build_3d_space方法构建三维空间")
            return None
        
        # 计算最小距离
        min_distances = self.calculate_min_distance_to_power_lines(distance_method)
        
        # 构建完整结果
        results = {
            'timestamp': str(np.datetime64('now')),
            'danger_objects_count': len(self.danger_objects),
            'power_lines_count': len(self.power_lines),
            'min_distances': min_distances,
            'scale_factor_used': self.scale_factor
        }
        
        return results
    
    def load_power_line_regions(self, regions_json_path: str):
        """
        从JSON文件加载输电线区域划分信息
        
        Args:
            regions_json_path: 区域划分JSON文件路径
            
        Returns:
            List[Dict]: 区域信息列表，每个区域包含边界和参数
        """
        try:
            with open(regions_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            regions = []
            
            # 支持多种区域划分文件格式
            if 'regions' in data:
                # 格式1: 顶层包含regions字段
                for region_data in data['regions']:
                    region = self._parse_region_data(region_data)
                    if region:
                        regions.append(region)
            elif isinstance(data, list):
                # 格式2: 直接是区域列表
                for region_data in data:
                    region = self._parse_region_data(region_data)
                    if region:
                        regions.append(region)
            else:
                # 格式3: 可能是单个区域或其他格式
                print(f"警告: 未知的区域划分文件格式，尝试作为单个区域处理")
                region = self._parse_region_data(data)
                if region:
                    regions.append(region)
            
            print(f"成功加载 {len(regions)} 个输电线区域")
            return regions
        except Exception as e:
            print(f"加载区域划分文件出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_region_data(self, region_data: Dict) -> Dict:
        """
        解析单个区域数据
        
        Args:
            region_data: 区域数据字典
            
        Returns:
            Dict: 标准化的区域信息字典
        """
        try:
            # 基本区域信息
            region = {
                'id': region_data.get('id', f"region_{id(region_data)}"),
                'label': region_data.get('label', 'power_line_region'),
                'smooth_params': region_data.get('smooth_params', {})
            }
            
            # 处理区域边界
            if 'polygon' in region_data:
                # 多边形边界
                region['type'] = 'polygon'
                region['polygon'] = region_data['polygon']
            elif 'bbox' in region_data:
                # 矩形边界 [x_min, y_min, x_max, y_max]
                region['type'] = 'bbox'
                region['bbox'] = region_data['bbox']
            elif 'points' in region_data and len(region_data['points']) >= 2:
                # 线区域，扩展为带状区域
                region['type'] = 'line_band'
                region['points'] = region_data['points']
                region['band_width'] = region_data.get('band_width', 20)  # 默认带宽20像素
            else:
                print(f"警告: 区域 {region['id']} 缺少有效的边界信息")
                return None
            
            # 验证区域数据
            if region['type'] == 'polygon' and not self._validate_polygon(region['polygon']):
                print(f"警告: 区域 {region['id']} 的多边形数据无效")
                return None
                
            # 提取区域特定的平滑参数
            region['smooth_params'] = {
                'sigma': region_data.get('sigma', region_data.get('global_sigma', 1.0)),
                'gradient_factor': region_data.get('gradient_factor', region_data.get('depth_gradient_factor', 0.3)),
                'direction_weight': region_data.get('direction_weight', 0.7),
                'apply_smooth': region_data.get('apply_smooth', True)
            }
            
            return region
        except Exception as e:
            print(f"解析区域数据时出错: {e}")
            return None
    
    def _validate_polygon(self, polygon: List[List[float]]) -> bool:
        """
        验证多边形数据是否有效
        
        Args:
            polygon: 多边形顶点列表 [[x1,y1], [x2,y2], ...]
            
        Returns:
            bool: 多边形数据是否有效
        """
        if not polygon or len(polygon) < 3:
            return False
        
        # 验证每个点都是有效的坐标对
        return all(isinstance(point, (list, tuple)) and len(point) >= 2 
                  and all(isinstance(coord, (int, float)) for coord in point[:2]) 
                  for point in polygon)
    
    def create_region_mask(self, region: Dict) -> np.ndarray:
        """
        根据区域信息创建掩码
        
        Args:
            region: 区域信息字典
            
        Returns:
            np.ndarray: 区域掩码，1表示在区域内，0表示在区域外
        """
        if self.depth_map is None:
            print("错误：请先设置深度图")
            return None
        
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        if region['type'] == 'polygon':
            # 创建多边形掩码
            from matplotlib.path import Path
            import matplotlib.patches as patches
            
            polygon = np.array(region['polygon'])
            x_coords, y_coords = polygon[:, 0], polygon[:, 1]
            
            # 创建网格坐标
            y_grid, x_grid = np.meshgrid(np.arange(self.image_height), np.arange(self.image_width), indexing='ij')
            grid_points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
            
            # 检查点是否在多边形内
            path = Path(polygon)
            inside = path.contains_points(grid_points)
            
            # 重塑回原始形状
            mask[inside.reshape((self.image_height, self.image_width))] = 1
            
        elif region['type'] == 'bbox':
            # 创建矩形掩码
            x_min, y_min, x_max, y_max = region['bbox']
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(self.image_width, int(x_max) + 1)
            y_max = min(self.image_height, int(y_max) + 1)
            
            mask[y_min:y_max, x_min:x_max] = 1
            
        elif region['type'] == 'line_band':
            # 创建线带状掩码
            points = region['points']
            band_width = region['band_width']
            
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # 计算线段方向和法线
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # 单位法线向量
                    nx, ny = -dy/length, dx/length
                    
                    # 计算带状区域的四个顶点
                    p1 = (x1 - nx * band_width/2, y1 - ny * band_width/2)
                    p2 = (x2 - nx * band_width/2, y2 - ny * band_width/2)
                    p3 = (x2 + nx * band_width/2, y2 + ny * band_width/2)
                    p4 = (x1 + nx * band_width/2, y1 + ny * band_width/2)
                    
                    # 创建多边形
                    band_polygon = [p1, p2, p3, p4]
                    
                    # 使用多边形方法填充带状区域
                    from matplotlib.path import Path
                    
                    polygon = np.array(band_polygon)
                    y_grid, x_grid = np.meshgrid(np.arange(self.image_height), np.arange(self.image_width), indexing='ij')
                    grid_points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
                    
                    path = Path(polygon)
                    inside = path.contains_points(grid_points)
                    
                    mask[inside.reshape((self.image_height, self.image_width))] = 1
        
        return mask
    
    def _is_3d_space_built(self) -> bool:
        """
        检查三维空间是否已构建
        
        Returns:
            bool: 是否已构建
        """
        # 放宽检查条件，只要有危险物和输电线就认为三维空间已构建
        # 即使部分3D坐标缺失，也允许继续计算距离
        if not self.danger_objects or not self.power_lines:
            print("警告: 缺少危险物或输电线数据")
            print(f"危险物数量: {len(self.danger_objects) if self.danger_objects else 0}")
            print(f"输电线数量: {len(self.power_lines) if self.power_lines else 0}")
            return False
        
        # 统计有3D坐标的危险物数量
        danger_with_3d = sum(1 for obj in self.danger_objects if obj.get('center_3d') is not None)
        
        # 统计有3D坐标的输电线数量
        lines_with_3d = 0
        points_with_3d = 0
        total_points = 0
        
        for line in self.power_lines:
            if 'points_3d' in line and line['points_3d']:
                lines_with_3d += 1
                for point_info in line['points_3d']:
                    total_points += 1
                    if point_info.get('3d') is not None:
                        points_with_3d += 1
        
        # 打印统计信息
        print(f"危险物3D坐标统计: 总数={len(self.danger_objects)}, 有3D坐标={danger_with_3d}")
        print(f"输电线3D坐标统计: 总数={len(self.power_lines)}, 有3D坐标={lines_with_3d}")
        print(f"输电线点3D坐标统计: 总数={total_points}, 有3D坐标={points_with_3d}")
        
        # 只要有部分3D坐标就允许继续计算
        return danger_with_3d > 0 and points_with_3d > 0
    
    def save_distance_results(self, output_path: str):
        """
        保存距离计算结果到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            results = self.calculate_all_distances()
            if results is None:
                return False
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"距离计算结果已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存距离计算结果出错: {e}")
            return False
    
    def get_3d_space_summary(self) -> Dict:
        """
        获取三维空间构建结果摘要
        
        Returns:
            Dict: 包含空间构建信息的字典
        """
        summary = {
            'danger_objects_count': len(self.danger_objects),
            'power_lines_count': len(self.power_lines),
            'has_depth_map': self.depth_map is not None,
            'has_scale_factor': self.scale_factor is not None,
            '3d_space_built': self._is_3d_space_built()
        }
        
        return summary
    
    def visualize_3d_point_cloud(self, output_path: str = None, show_plot: bool = False, show_all_points: bool = False, max_all_points: int = 10000) -> str:
        """
        可视化三维点云，包括危险物和输电线的3D位置
        
        Args:
            output_path: 输出图像路径，如果为None则不保存
            show_plot: 是否显示图像
            show_all_points: 是否显示图像中所有的三维点
            max_all_points: 当show_all_points为True时，最大显示点的数量（用于性能优化）
            
        Returns:
            保存的图像路径，如果没有保存则返回None
        """
        try:
            # 检查三维空间是否已构建
            if not self._is_3d_space_built():
                print("错误：三维空间尚未构建，请先调用build_3d_space方法")
                return None
            
            # 创建3D图形
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 初始化背景点坐标列表
            all_x = []
            all_y = []
            all_z = []
            
            # 如果需要显示所有点
            if show_all_points and self.depth_map is not None:
                print(f"正在添加所有点云数据（最大{max_all_points}个点）...")
                
                # 获取所有有效深度点的坐标
                y_indices, x_indices = np.where(self.depth_map > 0)
                valid_depths = self.depth_map[y_indices, x_indices]
                
                # 随机采样点以控制数量
                if len(valid_depths) > max_all_points:
                    sample_indices = np.random.choice(len(valid_depths), max_all_points, replace=False)
                    y_indices = y_indices[sample_indices]
                    x_indices = x_indices[sample_indices]
                    valid_depths = valid_depths[sample_indices]
                
                # 转换为3D坐标
                for x, y, depth in zip(x_indices, y_indices, valid_depths):
                    try:
                        point_3d = self._convert_2d_to_3d(int(x), int(y), depth)
                        if point_3d:
                            x_3d, y_3d, z_3d = point_3d
                            # 不再需要z坐标翻转，因为深度值已转换为正数
                            # 保持原坐标系统：x向右，y向上，z向前（远离相机）
                            all_x.append(x_3d)
                            all_y.append(y_3d)
                            all_z.append(z_3d)
                    except Exception as e:
                        continue
                
                # 绘制所有点云（使用透明度和小尺寸）
                if all_x:
                    print(f"成功添加 {len(all_x)} 个背景点云")
                    ax.scatter(all_x, all_y, all_z, c='gray', s=1, alpha=0.1, marker='.', label='Background Points')
            
            # Set figure title
            ax.set_title('3D Space Point Cloud Visualization', fontsize=15)
            
            # Set axis labels: 符合常规三维空间表示
            ax.set_xlabel('X (meters)', fontsize=10)  # X-axis: 向右为正
            ax.set_ylabel('Y (meters)', fontsize=10)  # Y-axis: 向上为正
            ax.set_zlabel('Z (meters)', fontsize=10)  # Z-axis: 向前/远离相机为正
            
            # 绘制危险物3D点
            danger_x = []
            danger_y = []
            danger_z = []
            
            for obj in self.danger_objects:
                if obj.get('center_3d') is not None:
                    x, y, z = obj['center_3d']
                    
                    # 翻转Y坐标使y轴以0为中心对称，Z坐标已为正数（符合常规表示）
                    # 保持原坐标系统：x向右，y向上，z向前（远离相机）
                    danger_x.append(x)
                    danger_y.append(y)
                    danger_z.append(z)
                    
                    # Draw hazard point and add label with correct coordinate order
                    ax.scatter(x, y, z, c='red', s=100, marker='o', label=f'Hazard {obj["id"]}' if len(danger_x) <= 1 else "")
                    # Add text label
                    ax.text(x, y, z, f'Object{obj["id"]}', fontsize=8)
            
            # 绘制输电线3D点和线段
            line_colors = ['blue', 'green', 'purple', 'orange', 'brown']
            line_count = 0
            line_points = []  # 用于存储所有输电线的坐标，供自动调整范围使用
            
            for line_idx, line in enumerate(self.power_lines):
                if 'points_3d' in line and line['points_3d']:
                    line_color = line_colors[line_count % len(line_colors)]
                    line_count += 1
                    
                    line_x = []
                    line_y = []
                    line_z = []
                    valid_points = []
                    current_line_points = []  # 当前线路的点
                    
                    for point_info in line['points_3d']:
                        if point_info.get('3d') is not None:
                            x, y, z = point_info['3d']
                            
                            # 翻转Y坐标使y轴以0为中心对称，Z坐标已为正数（符合常规表示）
                            # 保持原坐标系统：x向右，y向上，z向前（远离相机）
                            line_x.append(x)
                            line_y.append(y)
                            line_z.append(z)
                            valid_points.append(point_info)
                            current_line_points.append((x, y, z))  # 添加到当前线路点列表
                    
                    # 将当前线路的点添加到总列表
                    if current_line_points:
                        line_points.append(current_line_points)
                    
                    # Draw power line segments with correct coordinate order
                    if len(line_x) >= 2:
                        ax.plot(line_x, line_y, line_z, color=line_color, linewidth=2, 
                                label=f'Power Line {line.get("id", line_idx)}' if line_count <= 1 else "")
                        
                        # Draw power line points
                        ax.scatter(line_x, line_y, line_z, c=line_color, s=30, marker='^')
            
            # 添加图例
            legend_handles = []
            
            # 添加背景点云图例
            if show_all_points:
                legend_handles.append(Line2D([0], [0], marker='.', color='gray', linestyle='', markersize=10, alpha=0.5, label='Background Points'))
            
            # 添加危险物图例
            if len(danger_x) > 0:
                legend_handles.append(Line2D([0], [0], marker='o', color='red', linestyle='', markersize=10, label='Hazard Objects'))
            
            # 添加输电线图例
            if line_count > 0:
                for i in range(min(line_count, len(line_colors))):
                    legend_handles.append(Line2D([0], [0], color=line_colors[i], linestyle='-', linewidth=2, label=f'Power Line {i}'))
            
            if legend_handles:
                ax.legend(handles=legend_handles, loc='best', fontsize=8)
            
            # 设置视图角度
            ax.view_init(elev=30, azim=45)
            
            # 自动调整坐标轴范围以适应所有数据
            # 获取所有点的坐标范围
            all_coords = []
            
            # 添加背景点坐标
            if show_all_points and all_x:
                all_coords.extend(zip(all_x, all_y, all_z))
            
            # 添加危险物坐标
            if danger_x:
                all_coords.extend(zip(danger_x, danger_y, danger_z))
            
            # 添加输电线坐标
            for line in line_points:
                all_coords.extend(line)
            
            if all_coords:
                # 计算各轴的最小和最大值
                x_coords, y_coords, z_coords = zip(*all_coords)
                
                # 设置坐标轴范围，留出10%的边距
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                z_min, z_max = min(z_coords), max(z_coords)
                
                # 计算边距
                x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
                y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                z_margin = (z_max - z_min) * 0.1 if z_max != z_min else 1
                
                # 设置坐标轴范围
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.set_zlim(z_min - z_margin, z_max + z_margin)
                
                print(f"自动调整坐标轴范围:")
                print(f"  X轴: {x_min - x_margin:.2f} 到 {x_max + x_margin:.2f}")
                print(f"  Y轴: {y_min - y_margin:.2f} 到 {y_max + y_margin:.2f}")
                print(f"  Z轴: {z_min - z_margin:.2f} 到 {z_max + z_margin:.2f}")
            else:
                # 如果没有数据，使用默认范围
                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.set_zlim(0, 200)
            
            # 保存图像
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"三维点云可视化已保存到: {output_path}")
                
            # 显示图像
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return output_path
            
        except Exception as e:
            print(f"三维点云可视化出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_combined_visualization(self, image_path: str, distance_results: Dict, 
                                     output_path: str = None, show_plot: bool = False, 
                                     show_all_points: bool = False, max_all_points: int = 10000) -> str:
        """
        创建组合可视化，包含原始图像、距离结果和三维点云
        
        Args:
            image_path: 原始图像路径
            distance_results: 距离计算结果
            output_path: 输出图像路径，如果为None则不保存
            show_plot: 是否显示图像
            
        Returns:
            保存的图像路径，如果没有保存则返回None
        """
        try:
            import cv2
            from matplotlib.gridspec import GridSpec
            
            # 读取原始图像
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 如果图像不存在，创建一个简单的测试图像
                image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            
            # 创建三维点云可视化的临时图像
            temp_3d_path = 'temp_3d_visualization.png'
            self.visualize_3d_point_cloud(temp_3d_path, show_plot=False, 
                                        show_all_points=show_all_points, max_all_points=max_all_points)
            
            # 读取三维点云图像
            if os.path.exists(temp_3d_path):
                point_cloud_image = cv2.imread(temp_3d_path)
                point_cloud_image = cv2.cvtColor(point_cloud_image, cv2.COLOR_BGR2RGB)
            else:
                # 如果三维点云图像不存在，创建一个简单的占位图像
                point_cloud_image = np.ones((512, 512, 3), dtype=np.uint8) * 220
            
            # 创建组合图像
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(1, 2, figure=fig)
            
            # First subplot: Original image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(image)
            ax1.set_title('Original Image', fontsize=12)
            ax1.axis('off')
            
            # 在原始图像上标注危险物和距离
            for result in distance_results.get('min_distances', []):
                if result.get('danger_center_2d') and result.get('distance'):
                    # 获取危险物ID
                    obj_id = result['danger_object_id']
                    
                    # 找到对应的危险物对象
                    danger_obj = None
                    for obj in self.danger_objects:
                        if str(obj['id']) == obj_id:
                            danger_obj = obj
                            break
                    
                    # 绘制危险物中心（红色空心圆）
                    x, y = result['danger_center_2d']
                    circle = plt.Circle((x, y), 5, color='red', fill=False, linewidth=2)
                    ax1.add_patch(circle)
                    
                    # 如果有最近点信息，绘制最近点（绿色实心圆）
                    if danger_obj and 'closest_point_2d' in danger_obj:
                        closest_x, closest_y = danger_obj['closest_point_2d']
                        
                        # 绘制最近点
                        circle = plt.Circle((closest_x, closest_y), 6, color='green', fill=True, linewidth=2)
                        ax1.add_patch(circle)
                        
                        # 添加最近点标记
                        ax1.text(closest_x + 12, closest_y - 12, '✓', color='white', 
                                 backgroundcolor='green', fontsize=10, fontweight='bold')
                        
                        # 绘制从边界框中心到最近点的连线
                        ax1.plot([x, closest_x], [y, closest_y], color='green', linewidth=1, linestyle='--')
                    
                    # 添加距离文本
                    distance = result['distance']
                    ax1.text(x + 10, y + 10, f'{distance:.2f}m', color='white', 
                             backgroundcolor='red', fontsize=8)
                    
                    # 如果有最近点信息，绘制相关内容
                    if result.get('closest_point_info'):
                        closest_info = result['closest_point_info']
                        
                        # 处理不同的数据结构
                        if 'segment_start' in closest_info and 'segment_end' in closest_info:
                            # 线段模式：绘制线段
                            start = closest_info['segment_start']
                            end = closest_info['segment_end']
                            
                            # 在2D图像上绘制线段的简化表示
                            ax1.plot([start[0], end[0]], [start[1], end[1]], color='blue', linewidth=1)
                        elif 'point_2d' in closest_info:
                            # 点模式：绘制最近点
                            closest_point = closest_info['point_2d']
                            
                            # 在2D图像上绘制最近点
                            circle = plt.Circle((closest_point[0], closest_point[1]), 3, color='blue', fill=True, linewidth=1)
                            ax1.add_patch(circle)
                            
                            # 绘制从危险物中心到最近点的连接线
                            ax1.plot([x, closest_point[0]], [y, closest_point[1]], color='blue', linewidth=1, linestyle='--')
            
            # 第二个子图：三维点云可视化
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(point_cloud_image)
            ax2.set_title('3D Point Cloud Visualization', fontsize=12)
            ax2.axis('off')
            
            # Add overall title
            plt.suptitle('Hazard and Power Line Distance Analysis with 3D Visualization', fontsize=16)
            plt.tight_layout()
            
            # 保存组合图像
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"组合可视化结果已保存到: {output_path}")
            
            # 显示图像
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            # 清理临时文件
            if os.path.exists(temp_3d_path):
                os.remove(temp_3d_path)
                
            return output_path
            
        except Exception as e:
            print(f"创建组合可视化出错: {e}")
            import traceback
            traceback.print_exc()
            return None