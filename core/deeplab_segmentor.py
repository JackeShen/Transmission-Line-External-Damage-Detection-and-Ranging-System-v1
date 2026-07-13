"""
DeepLab分割器类
封装DeepLab模型进行语义分割
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import sys

# 添加deeplab目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'deeplab'))
from deeplab.deeplab import DeeplabV3


class DeepLabSegmentor:
    """DeepLab分割器"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
        self.num_classes = 2  # 默认类别数，加载模型时会更新
        
    def load_model(self, model_path: str, num_classes: int = 2, 
                   backbone: str = "convnextv2", input_shape: List[int] = [512, 512],
                   downsample_factor: int = 16, cuda: bool = True) -> bool:
        """
        加载DeepLab模型
        
        Args:
            model_path: 模型文件路径
            num_classes: 类别数量（包括背景）
            backbone: 主干网络名称
            input_shape: 输入图像尺寸 [height, width]
            downsample_factor: 下采样倍数
            cuda: 是否使用CUDA
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(model_path):
                return False
            
            # 如果模型路径是相对路径，需要转换为绝对路径
            if not os.path.isabs(model_path):
                # 先检查原始路径是否存在
                if not os.path.exists(model_path):
                    # 如果是相对于deeplab目录的路径
                    deeplab_dir = os.path.join(os.path.dirname(__file__), 'deeplab')
                    abs_path = os.path.join(deeplab_dir, model_path)
                    if os.path.exists(abs_path):
                        model_path = abs_path
                # 如果原始路径存在，转换为绝对路径
                else:
                    model_path = os.path.abspath(model_path)
            
            self.model_path = model_path
            self.num_classes = num_classes
            
            # 初始化DeepLab模型
            self.model = DeeplabV3(
                model_path=model_path,
                num_classes=num_classes,
                backbone=backbone,
                input_shape=input_shape,
                downsample_factor=downsample_factor,
                cuda=cuda,
                mix_type=0  # 混合模式：0=原图与分割图混合
            )

            # 生成类别名称（如果没有提供，使用默认名称）
            if len(self.class_names) == 0:
                # 根据类别数量设置默认名称
                if num_classes == 2:
                    # 默认两个类别：背景和cable
                    self.class_names = [ "电线"]
                else:
                    # 其他情况使用通用命名
                    self.class_names = [f"Class_{i}" for i in range(num_classes)]
                    if num_classes > 0:
                        self.class_names[0] = "background"
            
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_class_names(self, class_names: List[str]):
        """设置类别名称列表"""
        self.class_names = class_names.copy()
    
    def segment(self, image_path: str, conf_threshold: float = 0.25,
                class_filter: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        对图像进行语义分割
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值（DeepLab不使用此参数，保留以兼容接口）
            class_filter: 要分割的类别名称列表（可选，DeepLab会分割所有类别）
            
        Returns:
            Tuple[分割叠加图, 统计信息字典]
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 读取图像
        image = Image.open(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 使用DeepLab进行分割
        segmented_image_pil = self.model.detect_image(image, count=False, name_classes=self.class_names)
        
        # 转换为numpy数组（OpenCV格式）
        segmented_image = cv2.cvtColor(np.array(segmented_image_pil), cv2.COLOR_RGB2BGR)
        
        # 获取分割mask（用于统计）
        mask = self._get_segmentation_mask(image)
        
        # 计算统计信息
        stats = self._calculate_stats(mask, class_filter)
        
        return segmented_image, stats
    
    def _get_segmentation_mask(self, image: Image.Image) -> np.ndarray:
        """
        获取分割mask（类别ID的numpy数组）
        
        Args:
            image: PIL图像
            
        Returns:
            np.ndarray: 分割mask，每个像素值为类别ID
        """
        # 使用get_miou_png方法获取分割结果
        mask_pil = self.model.get_miou_png(image)
        mask = np.array(mask_pil)
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            # 假设是单通道的伪彩色，取第一个通道
            mask = mask[:, :, 0] if mask.shape[2] > 0 else mask[:, :, 0]
        
        return mask
    
    def _calculate_stats(self, mask: np.ndarray, class_filter: List[str] = None) -> Dict:
        """
        计算分割统计信息
        
        Args:
            mask: 分割mask
            class_filter: 类别过滤列表
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'segments': [],
            'class_counts': {},
            'class_areas': {},
            'total_segments': 0,
            'average_area': 0
        }
        
        h, w = mask.shape
        total_pixels = h * w
        
        # 统计每个类别的像素数和面积
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            # 如果指定了类别过滤，跳过不在过滤列表中的类别
            if class_filter is not None and len(class_filter) > 0:
                if class_name not in class_filter:
                    continue
            
            # 计算该类别的像素数
            class_pixels = np.sum(mask == class_id)
            class_area = int(class_pixels)
            
            if class_area > 0:
                stats['class_counts'][class_name] = 1  # DeepLab是语义分割，每个类别只有一个区域
                stats['class_areas'][class_name] = class_area
                
                stats['segments'].append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': 1.0,  # DeepLab不提供置信度
                    'area': class_area,
                    'ratio': class_area / total_pixels if total_pixels > 0 else 0.0
                })
        
        stats['total_segments'] = len(stats['segments'])
        if stats['total_segments'] > 0:
            total_area = sum(seg.get('area', 0) for seg in stats['segments'])
            stats['average_area'] = total_area / stats['total_segments']
        
        return stats
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names.copy()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
    
    def predict_mask(self, image_path: str) -> np.ndarray:
        """
        预测分割mask（不进行可视化）
        
        Args:
            image_path: 图像路径
            
        Returns:
            np.ndarray: 分割mask，每个像素值为类别ID
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        image = Image.open(image_path)
        mask = self._get_segmentation_mask(image)
        return mask

