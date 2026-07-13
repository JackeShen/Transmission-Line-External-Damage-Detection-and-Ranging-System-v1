"""
YOLO目标检测器类
支持加载和运行YOLO模型进行目标检测
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import os


class YOLODetector:
    """YOLO目标检测器"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
        self.task = None
        
    def load_model(self, model_path: str) -> bool:
        """
            加载YOLO模型
            
            Args:
                model_path: 模型文件路径
                
            Returns:
                bool: 是否加载成功
        """
        try:
            if not os.path.exists(model_path):
                return False
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.task = getattr(self.model, 'task', None)
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = []
            
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def detect(self, image_path: str, conf_threshold: float = 0.25, 
               class_filter: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        对图像进行目标检测
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            class_filter: 要检测的类别名称列表，如果为None则检测所有类别
            
        Returns:
            Tuple[检测后的图像, 检测结果列表]
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_image = image.copy()
        
        # 如果指定了类别过滤，转换为类别ID列表
        class_ids = None
        if class_filter is not None and len(class_filter) > 0:
            class_ids = []
            for class_name in class_filter:
                if class_name in self.class_names:
                    class_ids.append(self.class_names.index(class_name))
                else:
                    print(f"警告: 类别 '{class_name}' 不在模型类别列表中")
        
        # 进行推理
        if class_ids is not None:
            # 使用类别过滤
            results = self.model(image, conf=conf_threshold, classes=class_ids)
        else:
            # 检测所有类别
            results = self.model(image, conf=conf_threshold)
        
        # 解析检测结果
        detections = []
        annotated_image = original_image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取类别和置信度
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # 获取类别名称
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                
                # 如果指定了类别过滤，再次检查（双重保险）
                if class_filter is not None and len(class_filter) > 0:
                    if class_name not in class_filter:
                        continue
                
                # 绘制检测框
                color = self._get_class_color(cls_id)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 保存检测结果
                detections.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return annotated_image, detections
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """为不同类别生成不同颜色"""
        colors = [
            (0, 255, 255),    # 青色
            (255, 0, 255),    # 洋红
            (255, 255, 0),    # 黄色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 0, 255),      # 红色
        ]
        return colors[class_id % len(colors)]
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names.copy()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def segment(self, image_path: str, conf_threshold: float = 0.25,
                class_filter: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        对图像进行目标分割
        Returns: (分割叠加图, 统计信息字典)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载分割模型")

        if getattr(self.model, 'task', None) != 'segment':
            raise ValueError("当前模型不是分割模型，请加载YOLO分割模型（如 *-seg.pt）")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_image = image.copy()

        class_ids = None
        if class_filter is not None and len(class_filter) > 0:
            class_ids = []
            for class_name in class_filter:
                if class_name in self.class_names:
                    class_ids.append(self.class_names.index(class_name))

        if class_ids is not None:
            results = self.model(image, conf=conf_threshold, classes=class_ids)
        else:
            results = self.model(image, conf=conf_threshold)

        segmented_image = original_image.copy()
        segments = []
        class_counts = {}
        class_areas = {}

        for result in results:
            boxes = result.boxes
            masks = getattr(result, 'masks', None)

            if masks is None or masks.data is None:
                continue

            masks_data = masks.data.cpu().numpy()

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"

                if class_filter is not None and len(class_filter) > 0:
                    if class_name not in class_filter:
                        continue

                mask_array = masks_data[idx] if idx < len(masks_data) else None
                mask_area = 0
                if mask_array is not None:
                    mask_bool = mask_array > 0.5
                    mask_area = int(mask_bool.sum())
                    if mask_area > 0:
                        color = self._get_class_color(cls_id)
                        colored_mask = np.zeros_like(segmented_image, dtype=np.uint8)
                        colored_mask[mask_bool] = color
                        segmented_image = cv2.addWeighted(segmented_image, 0.6, colored_mask, 0.4, 0)

                # 绘制轮廓
                color = self._get_class_color(cls_id)
                cv2.rectangle(segmented_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(segmented_image, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(segmented_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                segments.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'area': mask_area
                })

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                class_areas[class_name] = class_areas.get(class_name, 0) + mask_area

        total = len(segments)
        average_area = 0
        if total > 0:
            total_area = sum(seg.get('area', 0) for seg in segments)
            average_area = total_area / total if total_area > 0 else 0

        stats = {
            'segments': segments,
            'class_counts': class_counts,
            'class_areas': class_areas,
            'total_segments': total,
            'average_area': average_area
        }

        return segmented_image, stats

