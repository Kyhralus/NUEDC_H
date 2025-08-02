#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Optional, Tuple
import collections

# ROS2 相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool
from cv_bridge import CvBridge

class TargetDetectionService(Node):
    """目标检测服务节点"""
    
    def __init__(self):
        super().__init__('target_detection_service')
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 初始化摄像头  
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        
        # 帧率计算相关
        self.fps_start_time = time.time()
        self.fps = 0
        
        # 透视变换相关
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        
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
        self.capture_thread = None
        
        # 性能统计
        self.timing_stats = collections.defaultdict(list)
        
        # 检测相关变量
        self.outer_rect = None
        self.inner_rect = None
        self.corners = None
        self.target_center = None
        self.target_circle = None
        self.target_circle_area = 0   # 近距离 50cm 左右 40000；远距离1.3m 左右 7800
        
        # 透视变换数据发布标志
        self.pub_perspective_data = False
        
        # 数据发布控制标志
        self.publish_target_data = False
        
        # 创建发布器
        self.target_publisher = self.create_publisher(
            String,
            'target_data',
            10
        )
        
        # 透视变换数据发布器
        self.warp_data_publisher = self.create_publisher(
            String,
            'warp_data',
            10
        )
        
        # 创建服务端
        self.perspective_service = self.create_service(
            SetBool,
            'set_perspective_publish',
            self.perspective_service_callback
        )
        
        # 创建目标数据发布控制服务
        self.detection_service = self.create_service(
            SetBool,
            'start_target_detection',
            self.detection_service_callback
        )
        
        # 启动摄像头和处理线程
        self.start_detection()
        
        self.get_logger().info('Target detection service started')
    
    def detection_service_callback(self, request, response):
        """处理目标数据发布控制服务的请求"""
        if request.data:
            # 启动数据发布
            self.publish_target_data = True
            response.success = True
            response.message = "目标数据发布已启动"
        else:
            # 停止数据发布
            self.publish_target_data = False
            response.success = True
            response.message = "目标数据发布已停止"
        
        self.get_logger().info(f"Received request to {('start' if request.data else 'stop')} target data publishing: {response.message}")
        return response
    
    def start_detection(self):
        """启动目标检测"""
        if self.is_running:
            return
            
        self.is_running = True
        self.frame_count = 0
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            self.is_running = False
            return
            
        # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置目标分辨率（1920x1080）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        
        # 验证设置是否成功
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera resolution set to: {actual_width}x{actual_height}")
        
        # 启动图像获取线程
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        # 创建定时器用于处理图像
        self.timer = self.create_timer(0.001, self.process_frame_from_queue)
        
        self.get_logger().info('Target detection started')
    
    def stop_detection(self):
        """停止目标检测"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # 释放摄像头资源
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        # 停止定时器
        if hasattr(self, 'timer'):
            self.destroy_timer(self.timer)
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.get_logger().info('Target detection stopped')
    
    def perspective_service_callback(self, request, response):
        """处理透视变换数据发布服务的请求"""
        self.pub_perspective_data = request.data
        response.success = True
        response.message = f"透视变换数据发布已设置为: {self.pub_perspective_data}"
        self.get_logger().info(f"Received request to set perspective data publish to: {self.pub_perspective_data}")
        return response
    
    def capture_frames(self):
        """图像捕获线程"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 帧计数增加
                self.frame_count += 1
                
                # 从图像中心截取960x720区域作为后续处理对象
                cropped_frame = self.crop_center_and_flip(frame, 960, 720)
                
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
        if not self.is_running:
            return
            
        try:
            frame = self.frame_queue.get_nowait()
            self.process_image(frame)
        except Empty:
            pass
    
    def crop_center_and_flip(self, image, target_width=960, target_height=720):
        """从图像中心裁剪指定尺寸区域，并进行上下翻转（高效实现）"""
        h, w = image.shape[:2]
        
        # 1. 计算中心裁剪的起始坐标
        start_x = max(0, (w - target_width) // 2)
        start_y = max(0, (h - target_height) // 2)
        
        # 2. 裁剪（直接切片操作，效率极高）
        end_x = start_x + min(target_width, w - start_x)
        end_y = start_y + min(target_height, h - start_y)
        cropped = image[start_y:end_y, start_x:end_x]  #  numpy切片，几乎不耗时
        
        # 3. 若尺寸不足则缩放（仅在必要时执行）
        if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
            cropped = cv2.resize(cropped, (target_width, target_height), 
                            interpolation=cv2.INTER_LINEAR)  # 线性插值速度快
        
        # 4. 上下翻转（OpenCV底层优化，耗时极短）
        flipped = cv2.flip(cropped, 0)  # flipCode=0 表示沿x轴翻转（上下翻转）
        
        return flipped
    
    
    def publish_perspective_matrix(self):
        """发布透视变换矩阵到ROS2话题"""
        if self.pub_perspective_data and self.perspective_matrix is not None:
            # 将矩阵展平为一维数组，并转换为字符串
            matrix_flat = self.perspective_matrix.flatten()
            matrix_str = ",".join(f"{x:.6f}" for x in matrix_flat)
            
            # 构建消息字符串，例如 "M:h11,h12,h13,h21,h22,h23,h31,h32,h33"
            msg_data = f"M:{matrix_str}"
            
            msg = String()
            msg.data = msg_data
            self.warp_data_publisher.publish(msg)
            self.get_logger().debug(f"Published perspective matrix: {msg_data}")

    def preprocess_image(self, frame):
        """优化的统一预处理步骤"""
        preprocess_start = time.time()
        
        # 一次性完成灰度化和高斯模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用高斯模糊替代双边滤波（更快）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        h, w = blurred.shape[:2]

        # 2. 提取中间640x480区域（若原图小于该尺寸则用全图）
        crop_w, crop_h = 640, 480
        # 计算中心区域坐标
        start_x = max(0, (w - crop_w) // 2)
        start_y = max(0, (h - crop_h) // 2)
        end_x = min(w, start_x + crop_w)
        end_y = min(h, start_y + crop_h)
        # 裁剪中心区域
        center_roi = blurred[start_y:end_y, start_x:end_x]
         # 3. 对中心区域计算Otsu阈值
        otsu_thresh, _ = cv2.threshold(center_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. 用该阈值对全图进行二值化
        _, binary = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
        # 可选：轻微形态学操作去除噪点
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 5. 边缘检测（基于二值化结果）
        edged = cv2.Canny(binary, 50, 150)  # Canny阈值可根据效果调整
        
        preprocess_time = (time.time() - preprocess_start) * 1000
        self.timing_stats['preprocess'].append(preprocess_time)
        
        return {
            'gray': gray,
            'blurred': blurred,
            'edged': edged
        }
    
    def detect_nested_rectangles_optimized(self, edged_image):
        """优化的嵌套矩形检测 - 增加稳定性并显示每步处理结果"""
        rect_start = time.time()
        
        # 1. 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        edged_stable = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)
        edged_stable = cv2.dilate(edged_stable, kernel, iterations=1)
        
        # 2. 轮廓检测
        contours, hierarchy = cv2.findContours(edged_stable, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. 筛选矩形
        rectangles = []
        filter_img = cv2.cvtColor(edged_stable, cv2.COLOR_GRAY2BGR)  # 用于显示筛选过程

        def calculate_angle(pt1, pt2, pt3):
            """计算由三个点构成的角的角度（pt2为顶点）"""
            # 向量pt2->pt1和pt2->pt3
            vec1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
            vec2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
            
            # 点积和模长
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            len1 = (vec1[0]**2 + vec1[1]** 2) **0.5
            len2 = (vec2[0]** 2 + vec2[1] ** 2) **0.5
            
            if len1 == 0 or len2 == 0:
                return 0.0  # 避免除以零
            
            # 计算夹角（弧度转角度）
            cos_theta = max(-1.0, min(1.0, dot_product / (len1 * len2)))  # 防止数值溢出
            angle = np.arccos(cos_theta) * (180 / np.pi)
            return angle

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # 面积筛选
            if area < 6000:
                cv2.drawContours(filter_img, [contour], -1, (128, 128, 128), 1)
                continue
            
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 顶点数和凸性筛选
            if not (4 <= len(approx) <= 6 and cv2.isContourConvex(approx)):
                cv2.drawContours(filter_img, [contour], -1, (0, 255, 255), 1)
                continue
            
            # 非四边形时拟合矩形（确保最终是4个顶点）
            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                approx = np.int0(box).reshape(-1, 1, 2)
            
            # 提取四个角点（按顺序排列，确保相邻性）
            corners = approx.reshape(4, 2)  # 转为4x2的角点列表
            
            # 检查四个角的角度是否在90°±25°范围内
            valid_corners = True
            for k in range(4):
                # 三个连续点：前一个点、当前点（顶点）、后一个点（循环取点）
                pt_prev = corners[(k - 1) % 4]
                pt_current = corners[k]
                pt_next = corners[(k + 1) % 4]
                
                angle = calculate_angle(pt_prev, pt_current, pt_next)
                
                # 角度不在65°~115°范围内，标记为无效
                if not (65 <= angle <= 115):
                    valid_corners = False
                    break
            
            if not valid_corners:
                # 用紫色绘制角度不符合的轮廓
                cv2.drawContours(filter_img, [approx], -1, (128, 0, 128), 1)
                continue
            
            # 长宽比筛选
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.8 < aspect_ratio < 1.5):
                cv2.drawContours(filter_img, [approx], -1, (0, 0, 255), 1)
                continue
            
            # 符合条件的矩形加入列表
            rectangles.append({
                'id': i,
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w // 2, y + h // 2),
                'corners': corners
            })
            # 用蓝色绘制通过筛选的矩形
            cv2.drawContours(filter_img, [approx], -1, (255, 0, 0), 2)

        # 4. 按面积排序
        rectangles.sort(key=lambda r: r['area'], reverse=True)

        # 5. 寻找嵌套矩形对
        outer_rect = inner_rect = None
        nested_img = cv2.cvtColor(edged_stable, cv2.COLOR_GRAY2BGR)

        for i, outer in enumerate(rectangles[:10]):
            x1, y1, w1, h1 = outer['bbox']
            cv2.rectangle(nested_img, (x1, y1), (x1 + w1, y1 + h1), (255, 165, 0), 2)
            
            for j, inner in enumerate(rectangles):
                if i == j:
                    continue
                x2, y2, w2, h2 = inner['bbox']
                
                # 嵌套条件判断
                margin = 5
                is_nested = (x1 <= x2 + margin and y1 <= y2 + margin and 
                            x1 + w1 >= x2 + w2 - margin and y1 + h1 >= y2 + h2 - margin)
                
                if is_nested:
                    area_ratio = inner['area'] / outer['area']
                    if 0.7 < area_ratio < 0.9:
                        outer_rect = outer
                        inner_rect = inner
                        cv2.rectangle(nested_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                        cv2.rectangle(nested_img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                        cv2.putText(nested_img, "Outer", (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(nested_img, "Inner", (x2, y2 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        break
            if outer_rect is not None:
                break
        
        # 计算耗时
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
        else:
            self.cache_miss_count += 1
        
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
            
            # 4. 渲染和发布
            render_start = time.time()
            result_image = self.render_results(
                cv_image, outer_rect, inner_rect, 
                target_center, target_circle
            )
            render_time = (time.time() - render_start) * 1000
            self.timing_stats['rendering'].append(render_time)
            
            # 5. 发布数据
            self.publish_detection_data(target_center, target_circle)
            
            # 6. 发布透视变换矩阵（如果开启）
            self.publish_perspective_matrix()
            
            # 7. 显示结果
            cv2.imshow('Target Detection Result', result_image)
            
            # 显示透视变换结果
            if perspective_vis is not None:
                cv2.imshow('Perspective Transform Visualization', perspective_vis)
            
            cv2.waitKey(1)
            
            # 8. 性能统计
            total_time = (time.time() - total_start_time) * 1000
            self.timing_stats['total'].append(total_time)
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            import traceback
            traceback.print_exc()

    def render_results(self, frame, outer_rect, inner_rect, target_center, target_circle):
        """渲染检测结果"""
        result_image = frame.copy()
        
        rect_detected = circle_detected = False
        
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
        
        # 添加状态信息
        cv2.putText(result_image, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(result_image, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        # 显示检测状态
        cv2.putText(result_image, f"Detection Status: {'Detected' if target_center else 'Not Detected'}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 目标圆面积
        cv2.putText(result_image, f"target circle area: {self.target_circle_area}", (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 状态显示
        if rect_detected and circle_detected:
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
    
    def publish_detection_data(self, target_center, target_circle):
        """发布检测数据"""
        # 只有当publish_target_data为True时才发布数据
        if not self.publish_target_data:
            return
            
        if target_center:
            if self.target_circle_area <= 9000: # 向上偏移15
                target_message = f"p,{target_center[0]},{target_center[1]}"
            elif self.target_circle_area >= 20000: # 向下偏移15
                target_message = f"p,{target_center[0]},{target_center[1]}"
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
        self.stop_detection()
        cv2.destroyAllWindows()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = TargetDetectionService()
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
