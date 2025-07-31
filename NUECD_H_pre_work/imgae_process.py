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

    def detect_circles_optimized(self, gray_image, binary_image, roi_bounds=None):
        """优化的圆形检测 - 支持ROI但保持640x480尺寸"""
        start_time = time.time()
        
        try:
            # 检查输入图像
            if binary_image is None:
                raise ValueError("二值图像为None")
            if gray_image is None:
                raise ValueError("灰度图像为None")
            
            # ROI处理 - 如果提供了ROI边界，只处理ROI区域
            if roi_bounds:
                x_min, y_min, x_max, y_max = roi_bounds
                # 确保边界有效
                height, width = binary_image.shape
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                
                # 提取ROI区域
                roi_binary = binary_image[y_min:y_max, x_min:x_max]
                roi_gray = gray_image[y_min:y_max, x_min:x_max]
                
                # 如果ROI太小，使用原图
                if roi_binary.size < 1000:
                    roi_binary = binary_image
                    roi_gray = gray_image
                    roi_offset = (0, 0)
                else:
                    roi_offset = (x_min, y_min)
                    # 将ROI调整为640x480尺寸
                    roi_binary = cv2.resize(roi_binary, (640, 480))
                    roi_gray = cv2.resize(roi_gray, (640, 480))
            else:
                # 没有ROI限制，直接使用原图并确保是640x480
                roi_binary = cv2.resize(binary_image, (640, 480))
                roi_gray = cv2.resize(gray_image, (640, 480))
                roi_offset = (0, 0)
                roi_bounds = (0, 0, binary_image.shape[1], binary_image.shape[0])
            
            # 查找轮廓
            contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(self.processing_times['circle_detection']) % 60 == 0:
                self.get_logger().info(f'圆形检测 - ROI尺寸: 640x480, 找到 {len(contours)} 个轮廓')
            
            circles = []
            filtered_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 快速面积过滤 - 根据640x480调整阈值
                min_area = self.circle_params['min_area'] * (640*480) / (binary_image.shape[0]*binary_image.shape[1])
                if area < min_area:
                    filtered_count += 1
                    continue
                
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # 半径过滤 - 根据640x480调整阈值
                scale_factor = 640 / binary_image.shape[1]
                min_radius = self.circle_params['min_radius'] * scale_factor
                max_radius = self.circle_params['max_radius'] * scale_factor
                
                if min_radius <= radius <= max_radius:
                    # 计算圆形度
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.7:  # 圆形度阈值
                            # 坐标映射回原图
                            if roi_bounds:
                                orig_scale_x = (roi_bounds[2] - roi_bounds[0]) / 640
                                orig_scale_y = (roi_bounds[3] - roi_bounds[1]) / 640
                                orig_x = int(x * orig_scale_x + roi_bounds[0])
                                orig_y = int(y * orig_scale_y + roi_bounds[1])
                                orig_radius = int(radius / scale_factor)
                            else:
                                orig_x, orig_y = int(x), int(y)
                                orig_radius = int(radius / scale_factor)
                            
                            circles.append({
                                'center': (orig_x, orig_y),
                                'radius': orig_radius,
                                'area': area / (scale_factor * scale_factor),  # 调整面积
                                'circularity': circularity,
                                'contour': contour
                            })
                else:
                    filtered_count += 1
            
            # 查找目标圆形
            target_center, target_circle = self.find_target_circles(circles)
            
            processing_time = time.time() - start_time
            self.processing_times['circle_detection'].append(processing_time)
            
            # 输出检测统计信息
            if len(self.processing_times['circle_detection']) % 60 == 0:
                self.get_logger().info(
                    f'圆形检测完成 - 耗时: {processing_time*1000:.1f}ms, '
                    f'有效圆形: {len(circles)}, 过滤掉: {filtered_count}'
                )
                
                if target_center:
                    self.get_logger().info(f'靶心检测到: {target_center}')
                if target_circle:
                    self.get_logger().info(f'目标圆检测到: 中心{target_circle["center"]}, 半径{target_circle["radius"]}')
            
            return {
                'circles': circles,
                'target_center': target_center,
                'target_circle': target_circle,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.get_logger().error(f'圆形检测出错: {str(e)}')
            import traceback
            self.get_logger().error(f'圆形检测错误详情: {traceback.format_exc()}')
            raise
    
    def find_target_circles(self, circles):
        """查找靶心和目标圆"""
        if not circles:
            return None, None
        
        # 找到面积最小的圆作为靶心
        target_center = None
        if circles:
            min_circle = min(circles, key=lambda c: c['area'])
            target_center = min_circle['center']
        
        # 查找目标半径范围内的圆形
        target_candidates = []
        for circle in circles:
            if (self.circle_params['target_radius_range'][0] <= 
                circle['radius'] <= self.circle_params['target_radius_range'][1]):
                target_candidates.append(circle)
        
        target_circle = None
        if target_candidates:
            if len(target_candidates) == 1:
                target_circle = target_candidates[0]
            else:
                # 多个候选圆，取平均值
                avg_x = int(np.mean([c['center'][0] for c in target_candidates]))
                avg_y = int(np.mean([c['center'][1] for c in target_candidates]))  
                avg_r = int(np.mean([c['radius'] for c in target_candidates]))
                avg_area = np.mean([c['area'] for c in target_candidates])
                avg_circularity = np.mean([c['circularity'] for c in target_candidates])
                
                target_circle = {
                    'center': (avg_x, avg_y),
                    'radius': avg_r,
                    'area': avg_area,
                    'circularity': avg_circularity,
                    'contour': target_candidates[0]['contour']  # 使用第一个的轮廓
                }
        
        return target_center, target_circle
    
    def process_detection_async(self, image_data):
        """异步处理检测任务 - 优化的ROI处理"""
        try:
            total_start_time = time.time()
            
            # 输出处理开始信息
            if len(self.processing_times['total']) % 120 == 0:
                self.get_logger().info('开始异步检测处理...')
            
            # 预处理
            preprocessed = self.preprocess_image(image_data['image'])
            
            # 矩形检测
            rect_result = self.detect_rectangles_optimized(preprocessed['binary'])
            
            # 根据矩形检测结果确定ROI
            roi_bounds = None
            if rect_result['nested_pair']:
                # 使用外矩形定义ROI，但保持合理的边界
                outer_rect = rect_result['nested_pair']['outer']
                x, y, w, h = outer_rect['bbox']
                
                # 扩展ROI边界
                margin = 30
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(preprocessed['binary'].shape[1], x + w + margin)
                y_max = min(preprocessed['binary'].shape[0], y + h + margin)
                
                roi_bounds = (x_min, y_min, x_max, y_max)
                
                self.get_logger().debug(f'使用ROI: ({x_min}, {y_min}, {x_max}, {y_max})')
            
            # 圆形检测 - 传入ROI边界
            circle_result = self.detect_circles_optimized(
                preprocessed['gray'], 
                preprocessed['binary'],
                roi_bounds
            )
            
            # 后处理
            start_time = time.time()
            detection_result = self.postprocess_results(
                preprocessed, rect_result, circle_result
            )
            postprocess_time = time.time() - start_time
            self.processing_times['postprocessing'].append(postprocess_time)
            
            # 总处理时间
            total_time = time.time() - total_start_time
            self.processing_times['total'].append(total_time)
            
            # 输出检测结果摘要
            if len(self.processing_times['total']) % 60 == 0:
                self.get_logger().info(
                    f'检测完成 - 总耗时: {total_time*1000:.1f}ms, '
                    f'状态: {detection_result["detection_status"]}, '
                    f'矩形数: {len(rect_result["rectangles"])}, '
                    f'圆形数: {len(circle_result["circles"])}, '
                    f'使用ROI: {"是" if roi_bounds else "否"}'
                )
            
            # 更新结果
            with self.detection_lock:
                self.last_detection_result = detection_result
            
            # 发布结果和显示
            self.publish_result(detection_result)
            if self.show_display:
                self.display_result(detection_result)
            
            # 打印性能统计
            self.print_performance_stats()
            
        except Exception as e:
            self.get_logger().error(f'检测处理出错: {str(e)}')
            import traceback
            self.get_logger().error(f'检测处理错误详情: {traceback.format_exc()}')
    
    def postprocess_results(self, preprocessed, rect_result, circle_result):
        """后处理检测结果 - 支持靶心和目标圆"""
        result = {
            'timestamp': time.time(),
            'image': preprocessed['original'],
            'rectangles': rect_result,
            'circles': circle_result,
            'detection_status': 'no_target',
            'target_center': None,
            'target_circle': None,
            'processing_times': {
                'preprocessing': preprocessed['processing_time'],
                'rectangle_detection': rect_result['processing_time'],
                'circle_detection': circle_result['processing_time']
            }
        }
        
        # 确定检测状态
        has_nested_rect = rect_result['nested_pair'] is not None
        has_target_center = circle_result['target_center'] is not None
        has_target_circle = circle_result['target_circle'] is not None
        
        if has_nested_rect and (has_target_center or has_target_circle):
            result['detection_status'] = 'target_detected'
            result['target_center'] = circle_result['target_center']
            result['target_circle'] = circle_result['target_circle']
        elif has_nested_rect:
            result['detection_status'] = 'rectangle_detected'
            result['target_center'] = rect_result['nested_pair']['inner']['center']
        
        return result
    
    def publish_result(self, result):
        """发布检测结果 - 支持靶心和目标圆分别发布"""
        try:
            messages_published = 0
            
            # 发布靶心数据
            if result['target_center']:
                x, y = result['target_center']
                data = f"p,{x},{y}"
                
                msg = String()
                msg.data = data
                self.target_publisher.publish(msg)
                messages_published += 1
                
                self.get_logger().info(f'✓ 发布靶心: {data}')
            
            # 发布目标圆数据
            if result['target_circle']:
                circle = result['target_circle']
                x, y = circle['center']
                r = circle['radius']
                data = f"c,{x},{y},{r}"
                
                msg = String()
                msg.data = data
                self.target_publisher.publish(msg)
                messages_published += 1
                
                self.get_logger().info(f'✓ 发布目标圆: {data}')
            
            # 状态信息
            if messages_published == 0:
                if len(self.processing_times['total']) % 120 == 0:  # 每120帧输出一次
                    self.get_logger().info(f'○ 未检测到目标 | 状态: {result["detection_status"]}')
            else:
                self.get_logger().info(f'总共发布 {messages_published} 条消息 | 状态: {result["detection_status"]}')
                    
        except Exception as e:
            self.get_logger().error(f'发布结果出错: {str(e)}')
            import traceback  
            self.get_logger().error(f'发布错误详情: {traceback.format_exc()}')
    
    def display_result(self, result):
        """显示检测结果 - 区分靶心和目标圆"""
        display_img = result['image'].copy()
        
        # 绘制矩形
        if result['rectangles']['nested_pair']:
            outer = result['rectangles']['nested_pair']['outer']
            inner = result['rectangles']['nested_pair']['inner']
            
            # 外矩形 - 绿色
            cv2.drawContours(display_img, [outer['contour']], -1, (0, 255, 0), 2)
            cv2.putText(display_img, "OUTER", 
                       (outer['center'][0]-30, outer['center'][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 内矩形 - 蓝色
            cv2.drawContours(display_img, [inner['contour']], -1, (255, 0, 0), 2)
            cv2.putText(display_img, "INNER", 
                       (inner['center'][0]-30, inner['center'][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制靶心 - 红色
        if result['target_center']:
            center = result['target_center']
            cv2.circle(display_img, center, 8, (0, 0, 255), 2)
            cv2.circle(display_img, center, 3, (0, 0, 255), -1)
            cv2.putText(display_img, f"靶心: {center}", 
                       (center[0]+10, center[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 绘制目标圆 - 绿色
        if result['target_circle']:
            circle = result['target_circle']
            center = circle['center']
            radius = circle['radius']
            cv2.circle(display_img, center, radius, (0, 255, 0), 2)
            cv2.circle(display_img, center, 5, (0, 255, 0), -1)
            cv2.putText(display_img, f"目标圆: ({center[0]}, {center[1]}), R={radius}", 
                       (center[0]-80, center[1]+radius+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 显示状态信息
        status_text = {
            'target_detected': ('Target Detected', (0, 255, 0)),
            'rectangle_detected': ('Rectangle Detected', (0, 255, 255)),
            'no_target': ('No Target', (0, 0, 255))
        }
        
        text, color = status_text[result['detection_status']]
        cv2.putText(display_img, text, (display_img.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示FPS
        fps_text = f'FPS: {self.fps_counter:.1f}'
        cv2.putText(display_img, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示ROI信息
        cv2.putText(display_img, 'ROI: 640x480', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Target Detection Result', display_img)
        cv2.waitKey(1)

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