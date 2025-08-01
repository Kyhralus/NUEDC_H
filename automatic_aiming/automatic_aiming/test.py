#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any

class OptimizedTargetDetector:
    """优化的目标检测器 - 基于物理参数约束"""
    
    def __init__(self):
        # 物理参数 (单位: cm)
        self.INNER_RECT_WIDTH = 25.5
        self.INNER_RECT_HEIGHT = 17.5
        self.OUTER_RECT_WIDTH = 29.1
        self.OUTER_RECT_HEIGHT = 21.1
        
        # 靶环半径 (单位: cm)
        self.TARGET_RADII = [2.0, 4.0, 6.0, 8.0]
        self.TARGET_CIRCLE_RADIUS = 6.0  # 目标圆半径
        
        # 图像参数
        self.IMAGE_WIDTH = 800
        self.IMAGE_HEIGHT = 600
        self.IMAGE_CENTER = (400, 300)
        
        # 计算尺寸比例约束
        self.rect_aspect_ratio = self.INNER_RECT_WIDTH / self.INNER_RECT_HEIGHT  # ≈ 1.46
        self.outer_inner_ratio = self.OUTER_RECT_WIDTH / self.INNER_RECT_WIDTH   # ≈ 1.14
        
        # 检测参数优化
        self.min_rect_area = 3000   # 最小矩形面积
        self.max_rect_area = 200000 # 最大矩形面积
        self.aspect_ratio_tolerance = 0.3  # 长宽比容差
        
        # 圆检测参数
        self.min_circle_radius = 10
        self.max_circle_radius = 100
        
        # 缓存变量，避免重复计算
        self._gray_cache = None
        self._blur_cache = None
        self._edges_cache = None
        
    def detect_target(self, image: np.ndarray) -> Dict[str, Any]:
        """
        主检测函数 - 优化版本
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果字典
        """
        if image is None or image.size == 0:
            return self._empty_result()
            
        # 清除缓存
        self._clear_cache()
        
        # 预处理图像（一次性完成所有预处理）
        processed_images = self._preprocess_image(image)
        
        # 1. 检测矩形框（优先级最高，精度要求最严格）
        outer_rect, inner_rect = self._detect_rectangles_optimized(
            processed_images['gray'], 
            processed_images['edges']
        )
        
        if outer_rect is None or inner_rect is None:
            return self._empty_result()
            
        # 2. 基于矩形框约束检测圆形
        target_center, target_circle = self._detect_circles_in_rect(
            processed_images['gray'],
            inner_rect
        )
        
        return {
            'outer_rect': outer_rect,
            'inner_rect': inner_rect,
            'target_center': target_center,
            'target_circle': target_circle,
            'success': target_center is not None
        }
    
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """一次性完成所有图像预处理，避免重复操作"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        
        # 形态学操作优化边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 缓存结果
        self._gray_cache = gray
        self._blur_cache = blur
        self._edges_cache = edges
        
        return {
            'gray': gray,
            'blur': blur,
            'edges': edges
        }
    
    def _detect_rectangles_optimized(self, gray: np.ndarray, edges: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        优化的矩形检测 - 基于物理参数约束
        """
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # 预筛选：基于面积和长宽比快速过滤
        candidate_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_rect_area <= area <= self.max_rect_area:
                # 快速长宽比检查
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    expected_ratio = self.rect_aspect_ratio
                    if abs(aspect_ratio - expected_ratio) / expected_ratio <= self.aspect_ratio_tolerance:
                        candidate_contours.append((contour, area, rect))
        
        if len(candidate_contours) < 2:
            return None, None
            
        # 按面积排序，大的是外框，小的是内框
        candidate_contours.sort(key=lambda x: x[1], reverse=True)
        
        # 验证外框和内框的尺寸关系
        for i in range(len(candidate_contours) - 1):
            outer_contour, outer_area, outer_rect = candidate_contours[i]
            
            for j in range(i + 1, len(candidate_contours)):
                inner_contour, inner_area, inner_rect = candidate_contours[j]
                
                # 检查面积比例
                area_ratio = outer_area / inner_area
                expected_area_ratio = (self.outer_inner_ratio ** 2)
                
                if abs(area_ratio - expected_area_ratio) / expected_area_ratio <= 0.3:
                    # 检查位置关系（内框应该在外框内部）
                    if self._is_rect_inside(inner_rect, outer_rect):
                        # 转换为标准矩形格式
                        outer_box = cv2.boxPoints(outer_rect).astype(np.int32)
                        inner_box = cv2.boxPoints(inner_rect).astype(np.int32)
                        return outer_box, inner_box
        
        return None, None
    
    def _detect_circles_in_rect(self, gray: np.ndarray, inner_rect: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int]]]:
        """
        在矩形区域内检测圆形 - 减少搜索范围
        """
        if inner_rect is None:
            return None, None
            
        # 获取内框的边界矩形
        x, y, w, h = cv2.boundingRect(inner_rect)
        
        # 扩展搜索区域（防止边界效应）
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        
        # 提取感兴趣区域
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, None
            
        # 基于物理参数估算圆的大小范围
        rect_diagonal = math.sqrt(w*w + h*h)
        estimated_max_radius = int(rect_diagonal * 0.4)  # 基于靶环最大半径估算
        estimated_min_radius = int(rect_diagonal * 0.05)  # 基于靶环最小半径估算
        
        # 限制搜索范围
        min_radius = max(self.min_circle_radius, estimated_min_radius)
        max_radius = min(self.max_circle_radius, estimated_max_radius)
        
        # 霍夫圆检测 - 参数优化
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min_radius * 1.5),  # 基于最小半径设置最小距离
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is None:
            return None, None
            
        circles = np.round(circles[0, :]).astype("int")
        
        # 寻找最内层圆（靶心）
        target_center = None
        target_circle = None
        min_radius_found = float('inf')
        
        for (cx, cy, r) in circles:
            # 转换回原图坐标
            abs_cx = cx + x1
            abs_cy = cy + y1
            
            # 验证圆心是否在内框内
            if self._point_in_rect((abs_cx, abs_cy), inner_rect):
                if r < min_radius_found:
                    min_radius_found = r
                    target_center = (abs_cx, abs_cy)
                    target_circle = (abs_cx, abs_cy, r)
        
        return target_center, target_circle
    
    def _is_rect_inside(self, inner_rect: Tuple, outer_rect: Tuple) -> bool:
        """检查内矩形是否在外矩形内部"""
        inner_center = inner_rect[0]
        outer_center = outer_rect[0]
        
        # 简单的中心距离检查
        distance = math.sqrt((inner_center[0] - outer_center[0])**2 + 
                           (inner_center[1] - outer_center[1])**2)
        
        # 如果中心距离较小，认为内框在外框内
        return distance < 50  # 可调整阈值
    
    def _point_in_rect(self, point: Tuple[int, int], rect: np.ndarray) -> bool:
        """检查点是否在矩形内"""
        return cv2.pointPolygonTest(rect, point, False) >= 0
    
    def _clear_cache(self):
        """清除缓存"""
        self._gray_cache = None
        self._blur_cache = None
        self._edges_cache = None
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'outer_rect': None,
            'inner_rect': None,
            'target_center': None,
            'target_circle': None,
            'success': False
        }
    
    def visualize_result(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        # 绘制外框
        if result['outer_rect'] is not None:
            cv2.drawContours(vis_image, [result['outer_rect']], -1, (0, 255, 0), 2)
            cv2.putText(vis_image, "Outer", tuple(result['outer_rect'][0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制内框
        if result['inner_rect'] is not None:
            cv2.drawContours(vis_image, [result['inner_rect']], -1, (255, 0, 0), 2)
            cv2.putText(vis_image, "Inner", tuple(result['inner_rect'][0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制目标圆和靶心
        if result['target_circle'] is not None:
            cx, cy, r = result['target_circle']
            cv2.circle(vis_image, (cx, cy), r, (0, 0, 255), 2)
            
        if result['target_center'] is not None:
            cx, cy = result['target_center']
            cv2.circle(vis_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis_image, f"Target({cx},{cy})", (cx+10, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image


def main():
    """测试函数"""
    detector = OptimizedTargetDetector()
    
    # 测试用例
    cap = cv2.VideoCapture(0)  # 或者使用图片路径
    # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 设置目标分辨率（1920x1080）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 验证设置是否成功
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 从1920x1080图像中截取中心800x600区域
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        x1 = center_x - 400  # 800 // 2
        y1 = center_y - 300  # 600 // 2
        x2 = center_x + 400
        y2 = center_y + 300
        frame = frame[y1:y2, x1:x2]
        
        # 检测目标
        result = detector.detect_target(frame)
        
        # 可视化结果
        vis_frame = detector.visualize_result(frame, result)
        
        # 显示结果
        cv2.imshow('Optimized Target Detection', vis_frame)
        
        # 打印检测结果
        if result['success']:
            print(f"检测成功! 靶心位置: {result['target_center']}")
        else:
            print("未检测到目标")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()