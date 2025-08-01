# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Optional, Tuple
import collections

class TargetDetector:
    def __init__(self):
        # 性能统计
        self.timing_stats = collections.defaultdict(list)
        
        # 添加时间平滑机制
        self.last_target_center = None
        self.last_target_circle = None
        self.detection_confidence = {'circle': 0}
        self.max_confidence = 5  # 最大置信度
        
        # 仿射变换相关
        self.affine_matrix = None
        self.inverse_affine_matrix = None
        
        # 先验知识：内框25.5 X 17.5 cm，目标圆半径6cm
        self.inner_rect_width_cm = 25.5
        self.inner_rect_height_cm = 17.5
        self.target_circle_radius_cm = 6.0
    
    def get_logger(self):
        """简单的日志记录器替代"""
        class SimpleLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
            def error(self, msg):
                print(f"[ERROR] {msg}")
            def warning(self, msg):
                print(f"[WARNING] {msg}")
        return SimpleLogger()
    
    def sort_corners(self, corners):
        """对角点进行排序：左上、右上、右下、左下"""
        # 计算质心
        centroid = np.mean(corners, axis=0)
        
        # 按角度排序
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        sorted_corners = sorted(corners, key=angle_from_centroid)
        
        # 找到最左上角的点作为起始点
        top_left_idx = 0
        min_dist = float('inf')
        for i, corner in enumerate(sorted_corners):
            dist = corner[0] + corner[1]  # 到左上角(0,0)的曼哈顿距离
            if dist < min_dist:
                min_dist = dist
                top_left_idx = i
        
        # 重新排列，从左上角开始顺时针
        reordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
        
        return np.array(reordered)
    
    def compute_affine_transform(self, inner_rect):
        """计算从inner_rect角点到640x480的仿射变换矩阵"""
        if inner_rect is None:
            return None, None
            
        # 获取inner_rect的四个角点，按顺序排列
        corners = inner_rect['corners']
        
        # 对角点进行排序：左上、右上、右下、左下
        corners = self.sort_corners(corners)
        
        # 目标区域：640x480
        target_corners = np.float32([
            [0, 0],      # 左上
            [640, 0],    # 右上
            [640, 480],  # 右下
            [0, 480]     # 左下
        ])
        
        # 计算仿射变换矩阵（使用透视变换获得更好的效果）
        affine_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
        inverse_matrix = cv2.getPerspectiveTransform(target_corners, corners.astype(np.float32))
        
        return affine_matrix, inverse_matrix
    
    def detect_circles_with_affine_optimized(self, frame, inner_rect):
        """优化的仿射变换圆形检测 - 基于先验知识的精确检测"""
        circle_start = time.time()
        
        if inner_rect is None:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, None
        
        # 计算仿射变换矩阵
        affine_matrix, inverse_matrix = self.compute_affine_transform(inner_rect)
        if affine_matrix is None:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, None
        
        # 保存变换矩阵
        self.affine_matrix = affine_matrix
        self.inverse_affine_matrix = inverse_matrix
        
        # 应用仿射变换
        warped_image = cv2.warpPerspective(frame, affine_matrix, (640, 480))
        
        # 显示仿射变换结果
        cv2.imshow('仿射变换结果', warped_image)
        
        # 基于先验知识计算期望参数
        # 仿射变换后640x480对应实际25.5x17.5cm
        pixel_per_cm_x = 640 / self.inner_rect_width_cm  # 约25.1像素/cm
        pixel_per_cm_y = 480 / self.inner_rect_height_cm  # 约27.4像素/cm
        pixel_per_cm = (pixel_per_cm_x + pixel_per_cm_y) / 2  # 平均像素密度
        
        # 期望的目标圆半径（像素）
        expected_radius_pixels = int(self.target_circle_radius_cm * pixel_per_cm)  # 约152像素
        
        # 期望的目标中心（仿射变换后图像的正中心）
        expected_center = (320, 240)  # 640x480的中心
        
        try:
            self.get_logger().info(f"先验知识 - 期望圆心: {expected_center}, 期望半径: {expected_radius_pixels}px")
        except:
            print(f"先验知识 - 期望圆心: {expected_center}, 期望半径: {expected_radius_pixels}px")
        
        # 优化的图像处理流程
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
        
        # 改进的阈值处理 - 使用Otsu自动阈值
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 优化的形态学操作 - 修复断续轮廓
        # 1. 先用较大的椭圆核进行闭运算连接断续的轮廓
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. 再用小核去除噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # 3. 最后用中等核再次闭运算确保轮廓完整
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_final)
        
        # 轮廓检测
        contours, _ = cv2.findContours(final_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建显示图像
        circle_detection_display = cv2.cvtColor(final_processed, cv2.COLOR_GRAY2BGR)
        
        # 在显示图像上标记期望的圆心和圆
        cv2.circle(circle_detection_display, expected_center, 5, (255, 255, 0), -1)
        cv2.circle(circle_detection_display, expected_center, expected_radius_pixels, (255, 255, 0), 2)
        cv2.putText(circle_detection_display, f"期望圆 R={expected_radius_pixels}", 
                   (expected_center[0]-80, expected_center[1]-expected_radius_pixels-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if not contours:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, circle_detection_display
        
        # 基于先验知识的候选圆筛选
        candidate_circles = []
        
        # 根据先验知识设定搜索范围
        min_radius = int(expected_radius_pixels * 0.6)  # 期望半径的60%-140%
        max_radius = int(expected_radius_pixels * 1.4)
        center_search_radius = 80  # 期望中心的±80像素范围
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150 or area > 50000:  # 调整面积阈值
                continue
            
            # 计算轮廓的凸度和圆度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            
            solidity = area / hull_area
            if solidity < 0.7:  # 凸度阈值
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # 放宽圆度阈值
                continue
            
            # 使用最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 基于先验知识的位置和半径筛选
            center_distance = np.sqrt((x - expected_center[0])**2 + (y - expected_center[1])**2)
            if center_distance > center_search_radius:
                continue
                
            if radius < min_radius or radius > max_radius:
                continue
            
            # 计算轮廓与最小外接圆的匹配度
            circle_area = np.pi * radius * radius
            area_ratio = area / circle_area
            if area_ratio < 0.6:  # 面积比阈值
                continue
            
            # 计算与期望圆的匹配度评分
            radius_score = 1.0 - abs(radius - expected_radius_pixels) / expected_radius_pixels
            center_score = 1.0 - center_distance / center_search_radius
            circularity_score = circularity
            
            # 综合评分
            total_score = radius_score * 0.4 + center_score * 0.4 + circularity_score * 0.2
            
            candidate_circles.append([int(x), int(y), int(radius), int(area), circularity, total_score])
            
            # 在显示图像上绘制候选圆 - 确保坐标为整数
            center_x, center_y = int(x), int(y)
            cv2.circle(circle_detection_display, (center_x, center_y), int(radius), (0, 255, 255), 2)
            
            # 修复putText坐标类型错误
            text_x = max(0, center_x - 20)
            text_y = max(15, center_y + int(radius) + 15)
            cv2.putText(circle_detection_display, f'评分:{total_score:.2f}', 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        if not candidate_circles:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, circle_detection_display
        
        candidate_circles = np.array(candidate_circles)
        
        # 按综合评分排序，选择最佳圆作为靶心
        best_idx = np.argmax(candidate_circles[:, 5])  # 第5列是总评分
        best_circle = candidate_circles[best_idx]
        innerest_x, innerest_y, innerest_r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
        
        # 在仿射变换后的图像坐标系中的目标中心
        target_center_warped = (innerest_x, innerest_y)
        
        # 在显示图像上绘制靶心 - 确保坐标为整数
        cv2.circle(circle_detection_display, target_center_warped, 8, (0, 0, 255), -1)
        cv2.circle(circle_detection_display, target_center_warped, innerest_r, (0, 0, 255), 3)
        
        # 修复putText坐标类型错误
        text_x = max(0, innerest_x + 15)
        text_y = max(15, innerest_y - 15)
        cv2.putText(circle_detection_display, "靶心", 
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 筛选目标圆（根据先验知识的半径范围）
        target_radius_min = int(expected_radius_pixels * 0.8)
        target_radius_max = int(expected_radius_pixels * 1.2)
        target_circles = candidate_circles[
            (candidate_circles[:, 2] >= target_radius_min) & 
            (candidate_circles[:, 2] <= target_radius_max)
        ]
        
        # 最终目标圆
        target_circle_warped = None
        if len(target_circles) == 1:
            target_circle_warped = (int(target_circles[0][0]), int(target_circles[0][1]), int(target_circles[0][2]))
        elif len(target_circles) > 1:
            # 选择评分最好的
            best_target_idx = np.argmax(target_circles[:, 5])
            target_circle_warped = (int(target_circles[best_target_idx][0]), 
                                  int(target_circles[best_target_idx][1]), 
                                  int(target_circles[best_target_idx][2]))
        else:
            # 如果没有合适的目标圆，使用最佳圆
            target_circle_warped = (innerest_x, innerest_y, innerest_r)
        
        # 在显示图像上绘制目标圆
        if target_circle_warped:
            tc_x, tc_y, tc_r = target_circle_warped
            cv2.circle(circle_detection_display, (tc_x, tc_y), tc_r, (0, 255, 0), 3)
            
            # 修复putText坐标类型错误
            text_x = max(0, tc_x - 50)
            text_y = max(15, tc_y - tc_r - 15)
            cv2.putText(circle_detection_display, f"目标圆 R={tc_r}", 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 逆变换到原图坐标系
        target_center = None
        target_circle = None
        
        if target_center_warped and self.inverse_affine_matrix is not None:
            # 逆变换靶心坐标
            warped_point = np.array([[[float(target_center_warped[0]), float(target_center_warped[1])]]], dtype=np.float32)
            original_point = cv2.perspectiveTransform(warped_point, self.inverse_affine_matrix)
            target_center = (int(original_point[0][0][0]), int(original_point[0][0][1]))
        
        if target_circle_warped and self.inverse_affine_matrix is not None:
            # 逆变换目标圆
            tc_x, tc_y, tc_r = target_circle_warped
            warped_circle_point = np.array([[[float(tc_x), float(tc_y)]]], dtype=np.float32)
            original_circle_point = cv2.perspectiveTransform(warped_circle_point, self.inverse_affine_matrix)
            
            # 计算逆变换后的半径（近似）
            warped_radius_point = np.array([[[float(tc_x + tc_r), float(tc_y)]]], dtype=np.float32)
            original_radius_point = cv2.perspectiveTransform(warped_radius_point, self.inverse_affine_matrix)
            
            original_radius = int(np.linalg.norm(
                original_radius_point[0][0] - original_circle_point[0][0]
            ))
            
            target_circle = (
                int(original_circle_point[0][0][0]), 
                int(original_circle_point[0][0][1]), 
                original_radius
            )
        
        circle_time = (time.time() - circle_start) * 1000
        self.timing_stats['circle_detection'].append(circle_time)
        
        return target_center, target_circle, circle_detection_display