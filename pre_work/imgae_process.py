#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from collections import deque
import math

class EnhancedCircleDetector:
    def __init__(self):
        # 稳定性参数
        self.history_size = 7  # 增加历史帧数
        self.center_history = deque(maxlen=self.history_size)
        self.quality_history = deque(maxlen=self.history_size)
        self.stability_threshold = 15  # 中心点稳定性阈值
        
        # 检测参数
        self.min_area = 50  # 最小面积
        self.max_area = 10000  # 最大面积
        self.min_circularity = 0.65  # 进一步降低圆度要求
        self.duplicate_dist_thresh = 10  # 去重距离阈值
        self.duplicate_radius_thresh = 10  # 去重半径阈值
        
        # 质量评估参数
        self.min_quality_score = 0.6  # 最小质量分数
        self.quality_weight_circularity = 0.4
        self.quality_weight_area_ratio = 0.3
        self.quality_weight_stability = 0.3
        
        # 自适应参数
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        
    def calculate_quality_score(self, circle_info: Dict, frame_stability: float) -> float:
        """计算圆形检测的质量分数"""
        # 圆度分数 (0-1)
        circularity_score = min(circle_info['circularity'] / 1.0, 1.0)
        
        # 面积合理性分数 (0-1)
        area = circle_info['area']
        if area < self.min_area:
            area_score = 0
        elif area > self.max_area:
            area_score = max(0, 1 - (area - self.max_area) / self.max_area)
        else:
            # 面积在合理范围内，根据面积大小给分
            ideal_area = (self.min_area + self.max_area) / 2
            area_score = 1 - abs(area - ideal_area) / ideal_area
        
        # 稳定性分数 (0-1)
        stability_score = max(0, 1 - frame_stability / self.stability_threshold)
        
        # 综合质量分数
        quality_score = (
            self.quality_weight_circularity * circularity_score +
            self.quality_weight_area_ratio * area_score +
            self.quality_weight_stability * stability_score
        )
        
        return quality_score
    
    def preprocess_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """改进的图像预处理"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用双边滤波保持边缘的同时去噪
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # ROI处理
        height, width = bilateral.shape[:2]
        margin = 90
        x1, y1 = max(0, margin), max(0, margin)
        x2, y2 = min(width - margin, width), min(height - margin, height)
        
        # 确保ROI区域有效
        if x2 <= x1 or y2 <= y1:
            print("ROI区域无效，使用全图")
            roi = bilateral
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            roi = bilateral[y1:y2, x1:x2]
        
        # 自适应阈值处理
        roi_thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_c
        )
        
        # 创建全尺寸的阈值图像
        thresh_full = np.zeros_like(bilateral)
        thresh_full[y1:y2, x1:x2] = roi_thresh
        
        return bilateral, thresh_full
    
    def morphological_operations(self, thresh: np.ndarray) -> np.ndarray:
        """优化的形态学操作"""
        # 使用椭圆形核，更适合圆形轮廓
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_rect = np.ones((2, 2), np.uint8)
        
        # 先开运算去除小噪声
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_rect)
        
        # 再闭运算连接断开的轮廓
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_ellipse)
        
        return closed
    
    def filter_contours(self, contours: List, hierarchy: np.ndarray) -> List[Dict]:
        """改进的轮廓筛选"""
        candidate_circles = []
        valid_parents = set()
        
        def is_duplicate_circle(new_contour, existing_circles):
            (nx, ny), nr = cv2.minEnclosingCircle(new_contour)
            for c in existing_circles:
                (ex, ey), er = cv2.minEnclosingCircle(c['contour'])
                dist = np.sqrt((nx - ex)**2 + (ny - ey)**2)
                if dist < self.duplicate_dist_thresh and abs(nr - er) < self.duplicate_radius_thresh:
                    return True
            return False
        
        # 寻找有效父轮廓
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 200 or area > 50000:  # 添加最大面积限制
                continue
            
            perimeter = cv2.arcLength(contours[i], True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 添加凸性检查
            hull = cv2.convexHull(contours[i])
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            convexity = area / hull_area
            
            if circularity >= self.min_circularity and convexity > 0.8:
                valid_parents.add(i)
        
        # 收集子轮廓
        def add_valid_child_contours(current_idx):
            child_idx = hierarchy[current_idx][2]
            while child_idx != -1:
                area = cv2.contourArea(contours[child_idx])
                if area < self.min_area or area > self.max_area:
                    child_idx = hierarchy[child_idx][0]
                    continue
                    
                perimeter = cv2.arcLength(contours[child_idx], True)
                if perimeter == 0:
                    child_idx = hierarchy[child_idx][0]
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 添加凸性检查
                hull = cv2.convexHull(contours[child_idx])
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    child_idx = hierarchy[child_idx][0]
                    continue
                convexity = area / hull_area

                if circularity >= self.min_circularity and convexity > 0.75:
                    if not is_duplicate_circle(contours[child_idx], candidate_circles):
                        candidate_circles.append({
                            'index': child_idx,
                            'contour': contours[child_idx],
                            'area': area,
                            'circularity': circularity,
                            'convexity': convexity,
                            'parent_index': current_idx
                        })

                add_valid_child_contours(child_idx)
                child_idx = hierarchy[child_idx][0]

        for parent_idx in valid_parents:
            add_valid_child_contours(parent_idx)
        
        return candidate_circles
    
    def select_target_circle(self, candidate_circles: List[Dict], hierarchy: np.ndarray) -> Optional[Dict]:
        """改进的目标圆选择策略"""
        if not candidate_circles:
            return None
        
        # 计算层级
        circle_levels = {}
        for circle in candidate_circles:
            idx = circle['index']
            level = 0
            next_idx = idx
            while hierarchy[next_idx][3] != -1:
                next_idx = hierarchy[next_idx][3]
                level += 1
            circle_levels[idx] = level
            circle['level'] = level
        
        # 选择最深层级的轮廓
        max_level = max(circle_levels.values())
        deepest_circles = [c for c in candidate_circles if c['level'] == max_level]
        
        if not deepest_circles:
            return None
        
        # 在最深层级中，综合考虑面积、圆度和质量选择最佳目标
        best_circle = None
        best_score = 0
        
        for circle in deepest_circles:
            # 计算综合分数
            area_score = 1 / (1 + circle['area'] / 1000)  # 面积越小分数越高
            circularity_score = circle['circularity']
            convexity_score = circle['convexity']
            
            total_score = 0.4 * area_score + 0.4 * circularity_score + 0.2 * convexity_score
            
            if total_score > best_score:
                best_score = total_score
                best_circle = circle
        
        return best_circle
    
    def stabilize_center(self, current_center: Tuple[int, int], quality_score: float) -> Tuple[int, int]:
        """使用质量加权的中心点稳定化"""
        self.center_history.append(current_center)
        self.quality_history.append(quality_score)
        
        if len(self.center_history) < 3:
            return current_center
        
        # 使用质量分数作为权重
        centers = np.array(list(self.center_history))
        qualities = np.array(list(self.quality_history))
        
        # 归一化质量分数作为权重
        weights = qualities / qualities.sum() if qualities.sum() > 0 else np.ones_like(qualities) / len(qualities)
        
        # 计算加权平均中心点
        stable_center = np.average(centers, axis=0, weights=weights).astype(int)
        
        return tuple(stable_center)

def detect_deepest_inner_circle_enhanced(frame: np.ndarray, detector: EnhancedCircleDetector = None) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """增强版圆形检测函数"""
    if detector is None:
        detector = EnhancedCircleDetector()
    
    result_frame = frame.copy()
    # 步骤1: 图像预处理
    blurred, thresh_full = detector.preprocess_image(frame)
    cv2.imshow("Blur Image", blurred)
    cv2.imshow("Thresholded ROI", thresh_full)

    # 步骤2: 形态学操作
    thresh = detector.morphological_operations(thresh_full)
    cv2.imshow("Thresholded Image", thresh)

    # 步骤3: 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("未找到轮廓")
        return result_frame, None

    # 步骤4: 圆度筛选，记录所有候选圆
    candidate_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < detector.min_area or area > detector.max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < detector.min_circularity:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        candidate_circles.append({
            'contour': cnt,
            'area': area,
            'circularity': circularity,
            'center': center,
            'radius': radius
        })

    if not candidate_circles:
        print("未找到候选圆形")
        return result_frame, None

    # 步骤5: 找最小面积圆和半径在90-120的圆
    minist_circle = min(candidate_circles, key=lambda c: c['area'])
    range_circles = [c for c in candidate_circles if 90 <= c['radius'] <= 120]

    # 步骤6: 计算target_circle
    if range_circles:
        mean_x = int(np.mean([c['center'][0] for c in range_circles]))
        mean_y = int(np.mean([c['center'][1] for c in range_circles]))
        mean_r = np.mean([c['radius'] for c in range_circles])
        target_center = (mean_x, mean_y)
        target_radius = mean_r
    else:
        target_center = None
        target_radius = None

    # 步骤7: 可视化所有候选圆
    for c in candidate_circles:
        cv2.circle(result_frame, c['center'], int(c['radius']), (0, 255, 0), 2)
        cv2.circle(result_frame, c['center'], 2, (0, 255, 0), -1)

    # 步骤8: 可视化minist_circle（蓝色）
    cv2.circle(result_frame, minist_circle['center'], int(minist_circle['radius']), (255, 0, 0), 2)
    cv2.circle(result_frame, minist_circle['center'], 5, (255, 0, 0), -1)
    cv2.putText(result_frame, f"Minist: ({minist_circle['center'][0]}, {minist_circle['center'][1]})",
                (minist_circle['center'][0] - 80, minist_circle['center'][1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 步骤9: 可视化target_circle（红色）
    if target_center is not None:
        cv2.circle(result_frame, target_center, int(target_radius), (0, 0, 255), 2)
        cv2.circle(result_frame, target_center, 5, (0, 0, 255), -1)
        cv2.putText(result_frame, f"Target: ({target_center[0]}, {target_center[1]})",
                    (target_center[0] - 80, target_center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result_frame, target_center

if __name__ == "__main__":
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 使用默认摄像头，可以根据需要修改为其他索引或视频文件路径
    
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    
    # 设置摄像头参数（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 创建检测器实例
    detector = EnhancedCircleDetector()
    
    print("增强版圆形检测器已启动，按 'q' 键退出")
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 进行圆形检测
            result_frame, target_center = detect_deepest_inner_circle_enhanced(frame, detector)
            
            # 显示结果
            cv2.imshow("Circle Detection Result", result_frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                filename = f"detection_result_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"图像已保存为: {filename}")
    
    except KeyboardInterrupt:
        print("程序被中断")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")