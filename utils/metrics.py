"""
评价指标计算工具
- mAP (mean Average Precision) for 目标检测
- IoU / mIoU / Pixel Accuracy for 语义分割
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


# ============================================================
# 目标检测 mAP
# ============================================================
def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个边界框的 IoU (Intersection over Union)"""
    x1_i = max(bbox1[0], bbox2[0])
    y1_i = max(bbox1[1], bbox2[1])
    x2_i = min(bbox1[2], bbox2[2])
    y2_i = min(bbox1[3], bbox2[3])
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11点插值法计算 Average Precision"""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
        ap += p / 11.0
    return ap


def calculate_map(
    detections: List[Dict],
    ground_truth: Optional[List[Dict]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict:
    """
    计算 mAP 和各类别统计

    Args:
        detections: [{'class_id': int, 'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        ground_truth: 同上格式(无confidence) 或 None
        num_classes: 类别总数
        iou_threshold: IoU阈值

    Returns:
        {'map': float, 'class_aps': dict, 'class_stats': dict, 'class_counts': dict}
    """
    if ground_truth is None or len(ground_truth) == 0:
        class_counts = defaultdict(int)
        for det in detections:
            class_counts[det.get('class_name', f"cls_{det['class_id']}")] += 1
        return {
            'map': 0.0, 'class_aps': {}, 'class_stats': {},
            'class_counts': dict(class_counts),
            'total_detections': len(detections)
        }

    det_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    for det in detections:
        det_by_class[det['class_id']].append(det)
    for gt in ground_truth:
        gt_by_class[gt['class_id']].append(gt)

    class_aps = {}
    class_stats = {}

    for cid in range(num_classes):
        c_dets = sorted(det_by_class[cid], key=lambda x: x['confidence'], reverse=True)
        c_gts = gt_by_class[cid]

        if len(c_gts) == 0:
            class_aps[cid] = 0.0
            class_stats[cid] = {'tp': 0, 'fp': len(c_dets), 'fn': 0, 'precision': 0.0, 'recall': 0.0}
            continue

        tp = np.zeros(len(c_dets))
        fp = np.zeros(len(c_dets))
        gt_matched = [False] * len(c_gts)

        for i, det in enumerate(c_dets):
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(c_gts):
                if not gt_matched[j]:
                    iou = calculate_iou(det['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(c_gts)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

        ap = calculate_ap(recalls, precisions)
        class_aps[cid] = ap
        last_tp = int(tp_cum[-1]) if len(tp_cum) > 0 else 0
        last_fp = int(fp_cum[-1]) if len(fp_cum) > 0 else 0
        class_stats[cid] = {
            'tp': last_tp, 'fp': last_fp,
            'fn': len(c_gts) - last_tp,
            'precision': float(precisions[-1]) if len(precisions) > 0 else 0.0,
            'recall': float(recalls[-1]) if len(recalls) > 0 else 0.0,
            'ap': ap
        }

    map_val = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
    class_counts = defaultdict(int)
    for det in detections:
        class_counts[det.get('class_name', f"cls_{det['class_id']}")] += 1

    return {
        'map': map_val, 'class_aps': class_aps,
        'class_stats': class_stats, 'class_counts': dict(class_counts),
        'total_detections': len(detections)
    }


# ============================================================
# 语义分割 IoU / mIoU
# ============================================================
def calculate_seg_iou(pred_mask: np.ndarray, gt_mask: np.ndarray,
                      class_id: int, ignore_index: int = 255) -> float:
    """计算单类 IoU"""
    pred_bin = (pred_mask == class_id).astype(np.uint8)
    gt_bin = (gt_mask == class_id).astype(np.uint8)
    if ignore_index != 255:
        valid = (gt_mask != ignore_index)
        pred_bin = pred_bin * valid
        gt_bin = gt_bin * valid
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def calculate_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray,
                              ignore_index: int = 255) -> float:
    """像素准确率"""
    if pred_mask.shape != gt_mask.shape:
        return 0.0
    if ignore_index != 255:
        valid = (gt_mask != ignore_index)
        correct = np.sum((pred_mask == gt_mask) & valid)
        total = np.sum(valid)
    else:
        correct = np.sum(pred_mask == gt_mask)
        total = pred_mask.size
    return float(correct / total) if total > 0 else 0.0


def calculate_mean_iou(pred_mask: np.ndarray, gt_mask: np.ndarray,
                       num_classes: int, ignore_index: int = 255) -> Tuple[float, Dict[int, float]]:
    """计算 mIoU"""
    class_ious = {}
    valid_count = 0
    for cid in range(num_classes):
        iou = calculate_seg_iou(pred_mask, gt_mask, cid, ignore_index)
        class_ious[cid] = iou
        if np.any(gt_mask == cid):
            valid_count += 1
    miou = sum(class_ious[c] for c in range(num_classes) if np.any(gt_mask == c)) / valid_count if valid_count > 0 else 0.0
    return float(miou), class_ious


def calculate_all_seg_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                               num_classes: int, class_names: List[str],
                               ignore_index: int = 255) -> Dict:
    """计算全部分割评价指标"""
    miou, class_ious = calculate_mean_iou(pred_mask, gt_mask, num_classes, ignore_index)
    pa = calculate_pixel_accuracy(pred_mask, gt_mask, ignore_index)
    iou_detail = {}
    for cid, iou in class_ious.items():
        name = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
        iou_detail[name] = iou
    return {
        'mean_iou': miou, 'pixel_accuracy': pa,
        'class_ious': class_ious, 'class_ious_detail': iou_detail
    }
