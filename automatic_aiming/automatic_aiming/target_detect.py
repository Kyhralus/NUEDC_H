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
        
        # 多线程相关
        self.frame_queue = Queue(maxsize=2)  # 图像队列，限制大小避免内存堆积
        self.processing_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=3)  # 线程池
        
        # 性能统计
        self.timing_stats = collections.defaultdict(list)
        self.frame_count = 0
        
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
                    cropped_frame = self.crop_center_image(frame, 800, 600)
                    
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
    
    def crop_rectangle_with_padding(self, image, rect, padding=30):
        """根据检测到的矩形裁剪图像，并添加指定的边距"""
        if rect is None:
            return self.crop_center_image(image, 800, 600)
            
        # 获取矩形的边界框
        x, y, w, h = rect['bbox']
        
        # 添加边距
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)
        
        # 裁剪图像
        cropped = image[y_min:y_max, x_min:x_max]
        
        # 如果裁剪后的尺寸过小，回退到中心裁剪
        if cropped.shape[0] < 100 or cropped.shape[1] < 100:
            return self.crop_center_image(image, 800, 600)
            
        # 调整到固定尺寸以保持一致性
        cropped = cv2.resize(cropped, (800, 600))
        
        return cropped
    
    def preprocess_image(self, frame):
        """统一的图像预处理步骤"""
        preprocess_start = time.time()
        
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 双边滤波（用于圆形检测）
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 形态学处理（用于矩形检测）
        kernel = np.ones((5, 5), np.uint8)
        img_close = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
        
        # Canny边缘检测（用于矩形检测）
        edged = cv2.Canny(img_close, 50, 150)
        
        # HSV转换（用于激光检测）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        preprocess_time = (time.time() - preprocess_start) * 1000
        self.timing_stats['preprocess'].append(preprocess_time)
        
        return {
            'gray': gray,
            'bilateral': bilateral,
            'edged': edged,
            'hsv': hsv
        }
    
    def detect_blue_purple_laser(self, hsv_image, target_point=None, min_area=50, max_area=1500):
        """检测蓝紫色激光点（优化版）"""
        laser_start = time.time()
        
        h, w = hsv_image.shape[:2]
        reference_point = target_point if target_point is not None else (w // 2, h // 2)
        
        # 蓝紫色激光的HSV阈值
        lower_hsv = np.array([25, 3, 208])
        upper_hsv = np.array([179, 255, 255])
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laser_point = None
        best_score = -1
        
        # 筛选符合条件的轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            # 计算轮廓中心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            
            # 计算距离基准点的距离
            dx = x - reference_point[0]
            dy = y - reference_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # 归一化距离
            max_possible_distance = np.sqrt((w//2)**2 + (h//2)**2)
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
    
    def detect_nested_rectangles_optimized(self, edged_image):
        """优化的嵌套矩形检测"""
        rect_start = time.time()
        
        # 轮廓检测
        contours, _ = cv2.findContours(edged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选矩形
        rectangles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000:  # 提前过滤小面积
                continue
                
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
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
        
        # 寻找嵌套矩形对
        outer_rect = inner_rect = None
        
        for i, outer in enumerate(rectangles):
            x1, y1, w1, h1 = outer['bbox']
            for j, inner in enumerate(rectangles):
                if i == j:
                    continue
                x2, y2, w2, h2 = inner['bbox']
                is_nested = (x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2)
                if is_nested:
                    area_ratio = inner['area'] / outer['area']
                    if 0.6 < area_ratio < 0.9:
                        outer_rect = outer
                        inner_rect = inner
                        break
            if outer_rect is not None:
                break
        
        rect_time = (time.time() - rect_start) * 1000
        self.timing_stats['rectangle_detection'].append(rect_time)
        
        return outer_rect, inner_rect
    
    def detect_circles_optimized(self, bilateral_image):
        """优化的圆形检测"""
        circle_start = time.time()
        
        height, width = bilateral_image.shape[:2]
        margin = 100
        x1, y1 = max(0, margin), max(0, margin)
        x2, y2 = min(width - margin, width), min(height - margin, height)

        if x2 <= x1 or y2 <= y1:
            roi = bilateral_image
        else:
            roi = bilateral_image[y1:y2, x1:x2]

        # 自适应阈值处理
        roi_thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 将ROI结果映射回全图
        thresh_full = np.zeros_like(bilateral_image)
        thresh_full[y1:y2, x1:x2] = roi_thresh

        # 形态学操作
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_rect = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(thresh_full, cv2.MORPH_OPEN, kernel_rect)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_ellipse)

        # 轮廓检测
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None

        # 筛选候选圆
        candidate_circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200: 
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 30:
                continue
            candidate_circles.append([int(x), int(y), int(radius), int(area)])

        if not candidate_circles:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None

        candidate_circles = np.array(candidate_circles)

        # 找到最小面积圆作为靶心
        min_area_idx = np.argmin(candidate_circles[:, 3])
        innerest_circle = candidate_circles[min_area_idx]
        innerest_x, innerest_y, innerest_r, innerest_area = innerest_circle
        
        if innerest_r < 30:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None
            
        target_center = (int(innerest_x), int(innerest_y))

        # 筛选目标圆
        target_circles = []
        for circle in candidate_circles:
            x, y, r, area = circle
            if 90 <= r <= 120:
                target_circles.append([x, y, r])

        # 最终目标圆
        target_circle = None
        if len(target_circles) == 1:
            target_circle = tuple(target_circles[0])
        elif len(target_circles) > 1:
            target_circles = np.array(target_circles)
            avg_x = int(np.mean(target_circles[:, 0]))
            avg_y = int(np.mean(target_circles[:, 1]))
            avg_r = int(np.mean(target_circles[:, 2]))
            target_circle = (avg_x, avg_y, avg_r)

        circle_time = (time.time() - circle_start) * 1000
        self.timing_stats['circle_detection'].append(circle_time)
        
        return target_center, target_circle

    def process_image(self, cv_image):
        """处理图像的主函数（多线程优化版）"""
        total_start_time = time.time()
        self.frame_count += 1
        
        try:
            # 修复帧率计算逻辑
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:  # 每秒计算一次FPS
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = current_time
            
            # 1. 统一预处理
            processed_data = self.preprocess_image(cv_image)
            
            # 2. 并行检测任务
            futures = []
            
            # 提交矩形检测任务
            rect_future = self.thread_pool.submit(
                self.detect_nested_rectangles_optimized, 
                processed_data['edged']
            )
            futures.append(('rectangle', rect_future))
            
            # 提交圆形检测任务
            circle_future = self.thread_pool.submit(
                self.detect_circles_optimized, 
                processed_data['bilateral']
            )
            futures.append(('circle', circle_future))
            
            # 等待检测结果
            detection_results = {}
            for task_name, future in futures:
                try:
                    detection_results[task_name] = future.result(timeout=0.1)  # 100ms超时
                except Exception as e:
                    self.get_logger().warning(f"{task_name} detection timeout or failed: {e}")
                    detection_results[task_name] = None
            
            # 解析结果
            outer_rect, inner_rect = detection_results.get('rectangle', (None, None))
            target_center, target_circle = detection_results.get('circle', (None, None))
            
            # 更新矩形检测结果缓存（用于下一帧的智能裁剪）
            if outer_rect is not None:
                self.last_outer_rect = outer_rect
            
            # 3. 激光检测（仅当激光检测标志开启时执行）
            blue_laser_point = None
            if self.enable_laser_detection and target_center is not None:
                blue_laser_point = self.detect_blue_purple_laser(processed_data['hsv'], target_center)
            
            # 4. 绘制结果和发布数据
            render_start = time.time()
            result_image = self.render_results(
                cv_image, outer_rect, inner_rect, 
                target_center, target_circle, blue_laser_point
            )
            render_time = (time.time() - render_start) * 1000
            self.timing_stats['rendering'].append(render_time)
            
            # 5. 发布数据
            self.publish_detection_data(target_center, target_circle, blue_laser_point)
            
            # 6. 显示结果
            cv2.imshow('Target Detection Result', result_image)
            cv2.waitKey(1)
            
            # 7. 性能统计
            total_time = (time.time() - total_start_time) * 1000
            self.timing_stats['total'].append(total_time)
            
            # 每100帧打印一次统计信息
            if self.frame_count % 100 == 0:
                self.print_performance_stats()
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

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
            cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 3)
            cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 3)
            cv2.circle(result_image, outer_rect['center'], 5, (0, 0, 255), -1)
            cv2.circle(result_image, inner_rect['center'], 5, (0, 0, 255), -1)
            
            cv2.putText(result_image, "Outer Rect", 
                       (outer_rect['center'][0]-50, outer_rect['center'][1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, "Inner Rect", 
                       (inner_rect['center'][0]-50, inner_rect['center'][1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制圆形目标
        if target_center or target_circle:
            circle_detected = True
            
            if target_center:
                cv2.circle(result_image, target_center, 1, (0, 0, 255), 3)
                cv2.circle(result_image, target_center, 1, (0, 0, 255), -1)
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
                cv2.circle(result_image, (tc_x, tc_y), tc_r, (0, 255, 0), 3)
                cv2.circle(result_image, (tc_x, tc_y), 1, (0, 255, 0), -1)
                cv2.putText(result_image, f"Target Circle: ({tc_x}, {tc_y}), R={tc_r}", 
                           (tc_x-100, tc_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
        
        # 激光检测状态
        laser_status = "Laser Detection: ON" if self.enable_laser_detection else "Laser Detection: OFF"
        cv2.putText(result_image, laser_status, (10, 150), 
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
        if target_center:
            target_message = f"p,{target_center[0]},{target_center[1]}"
            if target_circle:
                tc_x, tc_y, tc_r = target_circle
                target_message = f"c,{tc_x},{tc_y},{tc_r}"
            
            msg_pub = String()
            msg_pub.data = target_message
            self.target_publisher.publish(msg_pub)
        
        if blue_laser_point and target_center and self.enable_laser_detection:
            err_x = blue_laser_point[0] - target_center[0]
            err_y = blue_laser_point[1] - target_center[1]
            error_msg = String()
            error_msg.data = f"l,{err_x},{err_y}"
            self.target_publisher.publish(error_msg)

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
        self.get_logger().info("Node shutting down...")
    except Exception as e:
        self.get_logger().error(f"Node execution error: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()