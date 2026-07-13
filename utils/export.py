"""
结果导出工具 — CSV / TXT 格式
"""
import csv
import os
from typing import Dict, List


def export_detection_csv(file_path: str, stats: Dict):
    """导出检测结果到 CSV"""
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['指标', '值'])
        w.writerow(['mAP', f"{stats.get('map', 0):.4f}"])
        w.writerow(['总检测数', stats.get('total_detections', 0)])
        w.writerow([])
        w.writerow(['类别', '数量'])
        for cls_name, count in stats.get('class_counts', {}).items():
            w.writerow([cls_name, count])
        w.writerow([])
        w.writerow(['类别', '置信度', 'X1', 'Y1', 'X2', 'Y2'])
        for det in stats.get('detections', []):
            w.writerow([det['class_name'], f"{det['confidence']:.4f}",
                        det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3]])


def export_detection_txt(file_path: str, stats: Dict):
    """导出检测结果到 TXT"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("目标检测结果报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"mAP: {stats.get('map', 0):.4f}\n")
        f.write(f"总检测数: {stats.get('total_detections', 0)}\n\n")
        f.write("-" * 50 + "\n类别统计:\n" + "-" * 50 + "\n")
        for cls_name, count in stats.get('class_counts', {}).items():
            f.write(f"  {cls_name}: {count}\n")
        f.write("\n" + "-" * 50 + "\n检测详情:\n" + "-" * 50 + "\n")
        for i, det in enumerate(stats.get('detections', []), 1):
            f.write(f"\n[{i}] {det['class_name']}  conf={det['confidence']:.4f}\n")
            f.write(f"    bbox: ({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})\n")


def export_segmentation_csv(file_path: str, stats: Dict):
    """导出分割结果到 CSV"""
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['指标', '值'])
        w.writerow(['总实例数', stats.get('total_segments', 0)])
        w.writerow(['平均面积', f"{stats.get('average_area', 0):.2f}"])
        metrics = stats.get('metrics', {})
        if metrics:
            w.writerow(['mIoU', f"{metrics.get('mean_iou', 0):.4f}"])
            w.writerow(['Pixel Accuracy', f"{metrics.get('pixel_accuracy', 0):.4f}"])
        w.writerow([])
        w.writerow(['类别', '数量', '面积', 'IoU'])
        class_counts = stats.get('class_counts', {})
        class_areas = stats.get('class_areas', {})
        class_ious = metrics.get('class_ious_detail', {}) if metrics else {}
        for cls_name, count in class_counts.items():
            w.writerow([cls_name, count, f"{class_areas.get(cls_name, 0):.2f}",
                        f"{class_ious.get(cls_name, 0):.4f}"])


def export_segmentation_txt(file_path: str, stats: Dict):
    """导出分割结果到 TXT"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n导线分割结果报告\n" + "=" * 50 + "\n\n")
        f.write(f"总实例数: {stats.get('total_segments', 0)}\n")
        f.write(f"平均面积: {stats.get('average_area', 0):.2f}\n")
        metrics = stats.get('metrics', {})
        if metrics:
            f.write(f"mIoU: {metrics.get('mean_iou', 0):.4f}\n")
            f.write(f"Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f}\n")
        f.write("\n" + "-" * 50 + "\n类别统计:\n" + "-" * 50 + "\n")
        class_counts = stats.get('class_counts', {})
        class_areas = stats.get('class_areas', {})
        for cls_name, count in class_counts.items():
            f.write(f"  {cls_name}: {count} 个, 面积={class_areas.get(cls_name, 0):.0f}\n")


def export_distance_csv(file_path: str, results: Dict):
    """导出距离分析结果到 CSV"""
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['危险物ID', '类别', '最短距离(m)', '最近导线', '状态'])
        for d in results.get('min_distances', []):
            status = '⚠️ 危险' if d['distance'] < 5.0 else '✓ 安全'
            w.writerow([d['danger_object_id'], d.get('class_id', ''),
                        f"{d['distance']:.2f}", d['closest_line'], status])
