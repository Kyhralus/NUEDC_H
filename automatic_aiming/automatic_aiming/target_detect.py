# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import collections

# ROS2 相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class TargetDetectionNode(Node):
    """目标检测ROS2节点"""
    
    def __init__(self):
        super().__init__('target_detection_node')
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 初始化摄像头  
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            return
            
        # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置目标分辨率（1920x1080）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # 验证设置是否成功
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera resolution set to: {actual_width}x{actual_height}")
        
        self.target_publisher = self.create_publisher(
            String,
            '/target_data',
            10
        )
        
        # 帧率计算相关 - 修复逻辑
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
        # 激光寻找标志，默认关闭
        self.enable_laser_detection = False
        self.laser_min_area, self.laser_max_area = 60, 2000  # 激光检测面积范围
        
        # 透视变换相关（替换原来的仿射变换）
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.laser_expand_pixels = 30  # 可调参数：激光检测区域外扩像素数
        
        # 透视变换缓存机制
        self.last_corners = None              # 上一次的角点
        self.corner_change_threshold = 3.0    # 角点变化阈值（像素）
        self.cached_perspective_matrix = None # 缓存的透视变换矩阵
        self.cached_inverse_perspective_matrix = None # 缓存的逆透视变换矩阵
        self.cache_hit_count = 0              # 缓存命中次数
        self.cache_miss_count = 0             # 缓存未命中次数
        
        # 多线程相关
        self.frame_queue = Queue(maxsize=2)  # 图像队列，限制大小避免内存堆积
        self.processing_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=3)  # 线程池
        
        # 性能统计
        self.timing_stats = collections.defaultdict(list)
        self.frame_count = 0
        
        # 检测相关变量
        self.outer_rect = None
        self.inner_rect = None
        self.corners = None
        self.target_center = None
        self.target_circle = None
        self.target_circle_area = 0   # 近距离 50cm 左右 40000；远距离1.3m 左右 7800
        
        # 启动图像获取线程
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        # 创建定时器用于处理图像
        self.timer = self.create_timer(0.001, self.process_frame_from_queue)
        
        self.get_logger().info('Target detection node started')
        self.get_logger().info(f'Laser detection status: {"Enabled" if self.enable_laser_detection else "Disabled"}')
    
    def capture_frames(self):
        """图像捕获线程"""
        while True:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 从图像中心截取800x600区域作为后续处理对象
                    cropped_frame = self.crop_center_image(frame, 960, 500)
                    
                    # 非阻塞地放入队列
                    try:
                        self.frame_queue.put_nowait(cropped_frame)
                    except:
                        # 队列满时丢弃旧帧
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(cropped_frame)
                        except Empty:
                            pass
            time.sleep(0.001)  # 减少CPU占用
    
    def process_frame_from_queue(self):
        """从队列获取帧并处理"""
        try:
            frame = self.frame_queue.get_nowait()
            self.process_image(frame)
        except Empty:
            pass
    
    def crop_center_image(self, image, target_width=800, target_height=600):
        """从图像中心裁剪指定尺寸的区域"""
        h, w = image.shape[:2]
        
        # 计算裁剪区域的起始点
        start_x = max(0, (w - target_width) // 2)
        start_y = max(0, (h - target_height) // 2)
        
        # 计算实际裁剪尺寸（防止超出边界）
        actual_width = min(target_width, w - start_x)
        actual_height = min(target_height, h - start_y)
        
        # 裁剪图像
        cropped = image[start_y:start_y + actual_height, start_x:start_x + actual_width]
        
        # 如果裁剪后的尺寸不足目标尺寸，进行填充或缩放
        if cropped.shape[:2] != (target_height, target_width):
            cropped = cv2.resize(cropped, (target_width, target_height))
        
        return cropped
    
    
    def preprocess_image(self, frame):
        """优化的统一预处理步骤"""
        preprocess_start = time.time()
        
        # 一次性完成灰度化和高斯模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊替代双边滤波（更快）
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # 合并形态学处理和Canny边缘检测
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        edged = cv2.Canny(morphed, 40, 120)
        
        # 只在需要时转换HSV（激光检测开启时）
        hsv = None
        if self.enable_laser_detection:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        preprocess_time = (time.time() - preprocess_start) * 1000
        self.timing_stats['preprocess'].append(preprocess_time)
        
        return {
            'gray': gray,
            'blurred': blurred,
            'edged': edged,
            'hsv': hsv
        }
    
    def detect_nested_rectangles_optimized(self, edged_image):
        """优化的嵌套矩形检测 - 增加稳定性"""
        rect_start = time.time()
        
        # 对边缘图像进行额外的形态学操作以提高稳定性
        kernel = np.ones((2, 2), np.uint8)
        edged_stable = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)
        edged_stable = cv2.dilate(edged_stable, kernel, iterations=1)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edged_stable, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选矩形
        rectangles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 3000:  # 降低面积阈值以提高检测敏感度
                continue
                
            # 使用更宽松的多边形逼近
            epsilon = 0.02 * cv2.arcLength(contour, True)  # 从0.03降低到0.02
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 允许4-6个顶点的多边形，增加检测成功率
            if 4 <= len(approx) <= 6 and cv2.isContourConvex(approx):
                # 如果不是严格的四边形，尝试拟合矩形
                if len(approx) != 4:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    approx = np.int0(box).reshape(-1, 1, 2)
                
                x, y, w, h = cv2.boundingRect(approx)
                
                # 添加长宽比检查，但更宽松
                aspect_ratio = w / h if h > 0 else 0
                if 1.0 < aspect_ratio < 3.0:  # 允许更大的长宽比范围
                    rectangles.append({
                        'id': i,
                        'contour': contour,
                        'approx': approx,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w // 2, y + h // 2),
                        'corners': approx.reshape(4, 2)
                    })
        
        rectangles.sort(key=lambda r: r['area'], reverse=True)
        
        # 寻找嵌套矩形对 - 更宽松的条件
        outer_rect = inner_rect = None
        
        for i, outer in enumerate(rectangles[:10]):  # 只检查前10个最大的矩形
            x1, y1, w1, h1 = outer['bbox']
            for j, inner in enumerate(rectangles):
                if i == j:
                    continue
                x2, y2, w2, h2 = inner['bbox']
                
                # 更宽松的嵌套条件
                margin = 5  # 允许5像素的误差
                is_nested = (x1 <= x2 + margin and y1 <= y2 + margin and 
                           x1 + w1 >= x2 + w2 - margin and y1 + h1 >= y2 + h2 - margin)
                
                if is_nested:
                    area_ratio = inner['area'] / outer['area']
                    if 0.6 < area_ratio < 0.9:  # 更宽松的面积比
                        outer_rect = outer
                        inner_rect = inner
                        break
            if outer_rect is not None:
                break

        
        rect_time = (time.time() - rect_start) * 1000
        self.timing_stats['rectangle_detection'].append(rect_time)
        
        return outer_rect, inner_rect
    
    def corners_changed_significantly(self, new_corners):
        """检查角点是否发生显著变化"""
        if self.last_corners is None:
            return True
        
        # 计算每个角点的变化距离
        distances = []
        for i in range(4):
            dist = np.linalg.norm(new_corners[i] - self.last_corners[i])
            distances.append(dist)
        
        # 如果任何一个角点变化超过阈值，认为发生了显著变化
        max_change = max(distances)
        avg_change = np.mean(distances)
        
        # 使用更严格的判断条件
        significant_change = max_change > self.corner_change_threshold or avg_change > self.corner_change_threshold * 0.5
        
        if not significant_change:
            self.cache_hit_count += 1
            # self.get_logger().debug(f"透视变换缓存命中 - 最大变化: {max_change:.2f}px, 平均变化: {avg_change:.2f}px")
        else:
            self.cache_miss_count += 1
            # self.get_logger().debug(f"透视变换缓存未命中 - 最大变化: {max_change:.2f}px, 平均变化: {avg_change:.2f}px")
        
        return significant_change
    
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
    def compute_perspective_transform(self, inner_rect):
        """计算从inner_rect角点到640x480的透视变换矩阵（带缓存优化）"""
        if inner_rect is None:
            return None, None
            
        # 获取inner_rect的四个角点，按顺序排列
        corners = inner_rect['corners']
        
        # 对角点进行排序：左上、右上、右下、左下
        self.corners = self.sort_corners(corners)
        
        # 检查角点是否发生显著变化
        if not self.corners_changed_significantly(self.corners):
            # 使用缓存的透视变换矩阵
            if self.cached_perspective_matrix is not None and self.cached_inverse_perspective_matrix is not None:
                return self.cached_perspective_matrix, self.cached_inverse_perspective_matrix
        
        # 角点发生显著变化，重新计算透视变换矩阵
        # 目标区域：640x480
        target_corners = np.float32([
            [0, 0],      # 左上
            [640, 0],    # 右上
            [640, 480],  # 右下
            [0, 480]     # 左下
        ])
        
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(self.corners.astype(np.float32), target_corners)
        inverse_perspective_matrix = cv2.getPerspectiveTransform(target_corners, self.corners.astype(np.float32))
        
        # 更新缓存
        self.last_corners = self.corners.copy()
        self.cached_perspective_matrix = perspective_matrix
        self.cached_inverse_perspective_matrix = inverse_perspective_matrix
        
        return perspective_matrix, inverse_perspective_matrix
    
    def get_target_from_perspective(self, frame, inner_rect):
        """基于透视变换和先验知识直接确定靶心位置和目标圆
        不再检测圆，而是直接利用几何关系和先验知识确定目标
        """
        transform_start = time.time()
        
        if inner_rect is None:
            self.timing_stats['perspective_transform'].append(0)
            return None, None, None
        
        # 计算透视变换矩阵（带缓存优化）
        perspective_matrix, inverse_perspective_matrix = self.compute_perspective_transform(inner_rect)
        transform_time = (time.time() - transform_start) * 1000
        
        if perspective_matrix is None:
            self.timing_stats['perspective_transform'].append(transform_time)
            return None, None, None
        
        # 保存变换矩阵
        self.perspective_matrix = perspective_matrix
        self.inverse_perspective_matrix = inverse_perspective_matrix
        
        # 应用透视变换（用于可视化显示）
        warped_image = cv2.warpPerspective(frame, perspective_matrix, (640, 480))
        
        # 先验知识：透视变换后640x480对应实际25.5x17.5cm靶面
        physical_width_cm = 25.5
        physical_height_cm = 17.5
        target_circle_radius_cm = 6.0
        
        # 计算像素/厘米比例
        pixel_per_cm_x = 640 / physical_width_cm  # 约25.1像素/cm
        pixel_per_cm_y = 480 / physical_height_cm  # 约27.4像素/cm
        pixel_per_cm = (pixel_per_cm_x + pixel_per_cm_y) / 2  # 平均像素密度
        
        # 目标圆半径（像素）
        expected_radius_pixels = int(target_circle_radius_cm * pixel_per_cm)  # 约152像素
        
        # 先验知识：靶心位于变换后图像的中心
        target_center_warped = (320, 240)  # 640x480的正中心
        
        # 创建显示图像
        vis_image = np.copy(warped_image)
        
        # 在显示图像上绘制靶心和目标圆
        cv2.circle(vis_image, target_center_warped, 5, (0, 0, 255), -1)  # 靶心点
        cv2.circle(vis_image, target_center_warped, expected_radius_pixels, (0, 255, 0), 2)  # 目标圆
        
        # 添加文本标注
        cv2.putText(vis_image, "Target Center", 
                   (target_center_warped[0] + 10, target_center_warped[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis_image, f"R = {expected_radius_pixels}px ({target_circle_radius_cm}cm)", 
                   (target_center_warped[0] - 100, target_center_warped[1] + expected_radius_pixels + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加透视变换缓存信息到显示图像
        cache_info = f"Transform Cache: Hit={self.cache_hit_count}, Miss={self.cache_miss_count}, Time={transform_time:.2f}ms"
        cv2.putText(vis_image, cache_info, (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 透视逆变换回原图坐标系
        target_center = None
        target_circle = None
        
        if self.inverse_perspective_matrix is not None:
            # 透视逆变换靶心坐标
            warped_point = np.array([[[float(target_center_warped[0]), float(target_center_warped[1])]]], dtype=np.float32)
            original_point = cv2.perspectiveTransform(warped_point, self.inverse_perspective_matrix)
            target_center = (int(original_point[0][0][0]), int(original_point[0][0][1]))
            
            # 透视逆变换目标圆
            tc_x, tc_y = target_center_warped
            warped_circle_point = np.array([[[float(tc_x), float(tc_y)]]], dtype=np.float32)
            original_circle_point = cv2.perspectiveTransform(warped_circle_point, self.inverse_perspective_matrix)
            
            # 计算透视逆变换后的半径（近似）
            warped_radius_point = np.array([[[float(tc_x + expected_radius_pixels), float(tc_y)]]], dtype=np.float32)
            original_radius_point = cv2.perspectiveTransform(warped_radius_point, self.inverse_perspective_matrix)
            
            original_radius = int(np.linalg.norm(
                original_radius_point[0][0] - original_circle_point[0][0]
            ))
            
            target_circle = (
                int(original_circle_point[0][0][0]), 
                int(original_circle_point[0][0][1]), 
                original_radius
            )
            
            # 更新目标中心和圆形
            self.target_center = target_center
            self.target_circle = target_circle
            self.target_circle_area = 3.14*target_circle[2]*target_circle[2]
        
        self.timing_stats['perspective_transform'].append(transform_time)
        
        return target_center, target_circle, vis_image

    def process_image(self, cv_image):
        """优化的图像处理主函数"""
        total_start_time = time.time()
        self.frame_count += 1
        
        try:
            # 帧率计算
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = current_time
            
            # 1. 优化的预处理
            processed_data = self.preprocess_image(cv_image)
            
            # 2. 矩形检测
            rect_start = time.time()
            outer_rect, inner_rect = self.detect_nested_rectangles_optimized(processed_data['edged'])
            rect_time = (time.time() - rect_start) * 1000
            self.timing_stats['rectangle_detection'].append(rect_time)
            
            # 3. 根据透视变换和先验知识获取靶心位置和目标圆
            target_center, target_circle, perspective_vis = self.get_target_from_perspective(
                cv_image, inner_rect
            )
            
            # 4. 激光检测（仅当开启时）
            blue_laser_point = None
            if self.enable_laser_detection and outer_rect is not None and processed_data['hsv'] is not None:
                blue_laser_point = self.detect_blue_purple_laser(
                    processed_data['hsv'], target_center, self.laser_min_area, self.laser_max_area
                )
            
            # 5. 渲染和发布
            render_start = time.time()
            result_image = self.render_results(
                cv_image, outer_rect, inner_rect, 
                target_center, target_circle, blue_laser_point
            )
            render_time = (time.time() - render_start) * 1000
            self.timing_stats['rendering'].append(render_time)
            
            # 6. 发布数据
            self.publish_detection_data(target_center, target_circle, blue_laser_point)
            
            # 7. 显示结果
            cv2.imshow('Target Detection Result', result_image)
            
            # 显示透视变换结果
            if perspective_vis is not None:
                cv2.imshow('Perspective Transform Visualization', perspective_vis)
            
            cv2.waitKey(1)
            
            # 8. 性能统计
            total_time = (time.time() - total_start_time) * 1000
            self.timing_stats['total'].append(total_time)
            
            # 每100帧打印一次统计信息
            if self.frame_count % 100 == 0:
                self.print_performance_stats()
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            import traceback
            traceback.print_exc()

    def detect_blue_purple_laser(self, hsv_image, target_point=None, min_area=50, max_area=1500):
        """检测矩形区域的蓝紫色激光点检测（优化版）"""
        laser_start = time.time()
        
        h, w = hsv_image.shape[:2]
        reference_point = target_point if target_point is not None else (w // 2, h // 2)
        
        # 从self.corners计算包围矩形并扩展
        if hasattr(self, 'corners') and self.corners is not None and len(self.corners) >= 4:
            # 提取角点的x和y坐标（假设corners是形状为(4,2)的数组）
            x_coords = self.corners[:, 0]
            y_coords = self.corners[:, 1]
            
            # 计算原始包围矩形
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            
            # 扩展矩形：左右各扩10像素，向上扩15像素（向下不扩展）
            extend_x = 10
            extend_up = 15
            x = max(0, min_x - extend_x)  # 左边界不超出图像
            y = max(0, min_y - extend_up)  # 上边界不超出图像
            width = min(w - x, (max_x + extend_x) - x)  # 右边界不超出图像
            height = min(h - y, max_y - y)  # 下边界保持原始
            
            enclose_rect = (x, y, width, height)
            
            # 截取HSV图像中的对应区域
            x1, y1, w1, h1 = enclose_rect
            roi_hsv = hsv_image[y1:y1+h1, x1:x1+w1]
        else:
            # 如果没有有效的角点，使用全图
            roi_hsv = hsv_image
            enclose_rect = (0, 0, w, h)  # 全图作为默认区域
        
        # 蓝紫色激光的HSV阈值
        lower_hsv = np.array([13, 54, 149])
        upper_hsv = np.array([255, 164, 255])
        mask = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laser_point = None
        best_score = -1
        
        # 筛选符合条件的轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            # 计算轮廓中心（注意：坐标需要转换回原图坐标系）
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            # 加上ROI的偏移量，转换为原图坐标
            x_roi = int(M["m10"] / M["m00"])
            y_roi = int(M["m01"] / M["m00"])
            x = x_roi + enclose_rect[0]
            y = y_roi + enclose_rect[1]
            
            # 计算距离基准点的距离
            dx = x - reference_point[0]
            dy = y - reference_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # 归一化距离
            max_possible_distance = np.sqrt((w//2)**2 + (h//2)** 2)
            normalized_distance = distance / max_possible_distance
            
            # 计算评分
            area_score = area / max_area
            distance_score = 1 - normalized_distance
            score = 0.3 * area_score + 0.7 * distance_score
            
            # 更新最优激光点
            if score > best_score:
                best_score = score
                laser_point = (x, y)
        
        laser_time = (time.time() - laser_start) * 1000
        self.timing_stats['laser_detection'].append(laser_time)
        
        return laser_point


    def render_results(self, frame, outer_rect, inner_rect, target_center, target_circle, blue_laser_point):
        """渲染检测结果"""
        result_image = frame.copy()
        
        rect_detected = circle_detected = blue_laser_detected = False
        
        # 计算图像中心
        image_center_x = frame.shape[1] // 2
        image_center_y = frame.shape[0] // 2
        image_center = (image_center_x, image_center_y)
        
        # 绘制图像中心
        cv2.circle(result_image, image_center, 5, (255, 255, 255), -1)
        cv2.line(result_image, (image_center_x - 10, image_center_y), (image_center_x + 10, image_center_y), (255, 255, 255), 2)
        cv2.line(result_image, (image_center_x, image_center_y - 10), (image_center_x, image_center_y + 10), (255, 255, 255), 2)
        cv2.putText(result_image, "Image Center", (image_center_x + 15, image_center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 绘制矩形
        if outer_rect and inner_rect:
            rect_detected = True
            cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 2)
            cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 2)
            
            cv2.putText(result_image, "Outer Rect", 
                       (outer_rect['bbox'][0], outer_rect['bbox'][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, "Inner Rect", 
                       (inner_rect['bbox'][0], inner_rect['bbox'][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制圆形目标
        if target_center or target_circle:
            circle_detected = True
            
            if target_center:
                cv2.circle(result_image, target_center, 5, (0, 0, 255), -1)  # 靶心点
                cv2.putText(result_image, f"Target Center: ({target_center[0]}, {target_center[1]})", 
                           (target_center[0]+15, target_center[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 计算并显示目标中心与图像中心的偏差
                center_err_x = target_center[0] - image_center_x
                center_err_y = target_center[1] - image_center_y
                cv2.line(result_image, image_center, target_center, (0, 165, 255), 2)
                cv2.putText(result_image, f"Center Error: ({center_err_x}, {center_err_y})", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            if target_circle:
                tc_x, tc_y, tc_r = target_circle
                cv2.circle(result_image, (tc_x, tc_y), tc_r, (0, 255, 0), 2)  # 目标圆，线宽为2
        
        # 绘制激光点（仅当激光检测开启时）
        if blue_laser_point and self.enable_laser_detection:
            blue_laser_detected = True
            cv2.circle(result_image, blue_laser_point, 1, (255, 0, 0), 3)
            cv2.circle(result_image, blue_laser_point, 1, (255, 0, 0), -1)
            cv2.putText(result_image, f"Laser Point: ({blue_laser_point[0]}, {blue_laser_point[1]})", 
                       (blue_laser_point[0]+15, blue_laser_point[1]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if target_center:
                err_x = blue_laser_point[0] - target_center[0]
                err_y = blue_laser_point[1] - target_center[1]
                cv2.line(result_image, target_center, blue_laser_point, (255, 255, 0), 2)
                cv2.putText(result_image, f"Laser Error: ({err_x}, {err_y})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 添加状态信息
        cv2.putText(result_image, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(result_image, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        # 显示检测状态
        cv2.putText(result_image, f"Detection Status: {'Detected' if target_center else 'Not Detected'}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 激光检测状态
        laser_status = "Laser Detection: ON" if self.enable_laser_detection else "Laser Detection: OFF"
        cv2.putText(result_image, laser_status, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 目标圆面积
        cv2.putText(result_image, f"target circle area: {self.target_circle_area}", (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 状态显示
        if rect_detected and circle_detected and blue_laser_detected:
            status_text, status_color = "All Detected", (0, 255, 0)
        elif rect_detected and circle_detected:
            status_text, status_color = "Target Detected", (0, 255, 255)
        elif rect_detected:
            status_text, status_color = "Rectangle Detected", (0, 165, 255)
        else:
            status_text, status_color = "No Target", (0, 0, 255)
        
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = result_image.shape[1] - text_size[0] - 10
        text_y = 30
        
        cv2.rectangle(result_image, (text_x-5, text_y-25), (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
        cv2.putText(result_image, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return result_image
    
    def publish_detection_data(self, target_center, target_circle, blue_laser_point):
        """发布检测数据"""
        # if blue_laser_point and  self.enable_laser_detection:
        #     target_message = f"p,{blue_laser_point[0]},{blue_laser_point[1]}"
        #     msg_pub = String()
        #     msg_pub.data = target_message
        #     self.target_publisher.publish(msg_pub)
        if target_center:
            if self.target_circle_area <= 9000: # 向上偏移15
                target_message = f"p,{target_center[0]},{target_center[1]-15}"
            elif self.target_circle_area >= 20000: # 向下偏移15
                target_message = f"p,{target_center[0]},{target_center[1]+15}"
            else:
                target_message = f"p,{target_center[0]},{target_center[1]}"
            msg_pub = String()
            msg_pub.data = target_message
            self.target_publisher.publish(msg_pub)
        elif target_circle:
            tc_x, tc_y, tc_r = target_circle
            target_message = f"c,{tc_x},{tc_y},{tc_r}"
            msg_pub = String()
            msg_pub.data = target_message
            self.target_publisher.publish(msg_pub)

    def print_performance_stats(self):
        """打印性能统计信息"""
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Performance Stats (Recent 100 frames, Total frames: {self.frame_count})")
        self.get_logger().info("=" * 60)
        
        for task, times in self.timing_stats.items():
            if times:
                avg_time = np.mean(times[-100:])  # 最近100次的平均值
                max_time = np.max(times[-100:])
                min_time = np.min(times[-100:])
                self.get_logger().info(f"{task:20s}: Avg={avg_time:6.2f}ms, Max={max_time:6.2f}ms, Min={min_time:6.2f}ms")
        
        # 透视变换缓存统计
        total_transforms = self.cache_hit_count + self.cache_miss_count
        if total_transforms > 0:
            cache_hit_rate = self.cache_hit_count / total_transforms * 100
            self.get_logger().info(f"Perspective Transform Cache Hit Rate: {cache_hit_rate:.1f}% ({self.cache_hit_count}/{total_transforms})")
        
        self.get_logger().info(f"Current FPS: {self.fps:.1f}")
        self.get_logger().info("=" * 60)
        
        # 清理旧的统计数据，保持内存使用合理
        for task in self.timing_stats:
            if len(self.timing_stats[task]) > 200:
                self.timing_stats[task] = self.timing_stats[task][-100:]
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        cv2.destroyAllWindows()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = TargetDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node shutting down...")
    except Exception as e:
        node.get_logger().error(f"Node execution error: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()