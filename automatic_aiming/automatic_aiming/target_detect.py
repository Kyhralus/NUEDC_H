# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from typing import Optional, Tuple

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
        self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
            return
            
        # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置目标分辨率（1920x1080）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # 验证设置是否成功
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"摄像头分辨率设置为: {actual_width}x{actual_height}")
        
        self.target_publisher = self.create_publisher(
            String,
            '/target_data',
            10
        )
        
        # 帧率计算相关
        self.prev_time = time.time()
        self.fps = 0
        
        # 创建定时器用于读取摄像头
        self.timer = self.create_timer(0.001, self.camera_callback)  # 约30fps
        
        self.get_logger().info('目标检测节点已启动')
    
    def crop_center_image(self, image, target_width=640, target_height=480):
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
    
    def camera_callback(self):
        """摄像头回调函数"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 从1920x1080图像中心裁剪640x480区域
                cropped_frame = self.crop_center_image(frame, 640, 480)
                
                # 将OpenCV图像转换为ROS消息并发布
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(cropped_frame, "bgr8")
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    # 这里可以添加发布图像的逻辑，或者直接处理
                    self.process_image(cropped_frame)
                except Exception as e:
                    self.get_logger().error(f"图像转换错误: {str(e)}")
    
    def detect_blue_purple_laser(
        self,
        frame: np.ndarray, 
        target_point: Optional[tuple[int, int]],
        min_area: int = 50,
        max_area: int = 1500,
        erode_iter: int = 1,
        dilate_iter: int = 2
    ) -> tuple[Optional[tuple[int, int]], np.ndarray]:
        """检测蓝紫色激光点，优先选择靠近target_point的激光点"""
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        reference_point = target_point if target_point is not None else (w // 2, h // 2)
        
        # 蓝紫色激光的HSV阈值
        lower_hsv = np.array([117, 45, 159])
        upper_hsv = np.array([168, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=erode_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=dilate_iter)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laser_point = None
        best_score = -1
        
        # 筛选符合条件的轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 面积过滤
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
        
        return laser_point, result_frame

    def process_image(self, cv_image):
        """处理图像的主函数"""
        total_start_time = time.time()
        
        try:
            # 计算帧率
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self.prev_time)) if self.prev_time else 0
            self.prev_time = now
            
            # 1. 检测嵌套矩形（直接在640x480图像上检测）
            outer_rect, inner_rect, _, _ = self.detect_nested_rectangles(cv_image)
            
            result_image = cv_image.copy()
            rect_detected = False
            circle_detected = False
            blue_laser_detected = False
            target_message = None
            target_center = None
            target_circle = None
            
            # 画出检测到的矩形并打印信息
            if outer_rect is not None and inner_rect is not None:
                rect_detected = True
                
                # 绘制矩形框
                cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 3)
                cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 3)
                cv2.circle(result_image, outer_rect['center'], 5, (0, 0, 255), -1)
                cv2.circle(result_image, inner_rect['center'], 5, (0, 0, 255), -1)
                
                # 添加矩形标签
                cv2.putText(result_image, "OUTER RECT", 
                           (outer_rect['center'][0]-50, outer_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result_image, "INNER RECT", 
                           (inner_rect['center'][0]-50, inner_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 打印矩形详细信息
                print("=" * 50)
                print("矩形检测信息:")
                print(f"外矩形 - 角点: {outer_rect['corners'].tolist()}")
                print(f"外矩形 - bbox(x,y,w,h): {outer_rect['bbox']}")
                print(f"外矩形 - 中心点: {outer_rect['center']}")
                print(f"外矩形 - 面积: {outer_rect['area']}")
                print(f"内矩形 - 角点: {inner_rect['corners'].tolist()}")
                print(f"内矩形 - bbox(x,y,w,h): {inner_rect['bbox']}")
                print(f"内矩形 - 中心点: {inner_rect['center']}")
                print(f"内矩形 - 面积: {inner_rect['area']}")
            
            # 2. 检测圆形目标（直接在640x480图像上检测）
            circle_result, target_center, target_circle, target_message, _ = self.detect_circles(cv_image)
            
            if target_center is not None or target_circle is not None:
                circle_detected = True
                
                # 绘制靶心
                if target_center is not None:
                    cv2.circle(result_image, target_center, 8, (0, 0, 255), 3)
                    cv2.circle(result_image, target_center, 3, (0, 0, 255), -1)
                    cv2.putText(result_image, f"靶心: ({target_center[0]}, {target_center[1]})", 
                               (target_center[0]+15, target_center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 打印靶心信息
                    print("靶心检测信息:")
                    print(f"靶心坐标: {target_center}")
                
                # 绘制目标圆
                if target_circle is not None:
                    tc_x, tc_y, tc_r = target_circle
                    cv2.circle(result_image, (tc_x, tc_y), tc_r, (0, 255, 0), 3)
                    cv2.circle(result_image, (tc_x, tc_y), 5, (0, 255, 0), -1)
                    cv2.putText(result_image, f"目标圆: ({tc_x}, {tc_y}), R={tc_r}", 
                               (tc_x-100, tc_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 打印目标圆信息
                    print("目标圆检测信息:")
                    print(f"目标圆圆心坐标: ({tc_x}, {tc_y})")
                    print(f"目标圆半径: {tc_r}")
            
            # 3. 检测蓝紫色激光（直接在640x480图像上检测）
            blue_laser_point, _ = self.detect_blue_purple_laser(cv_image, target_center)
            if blue_laser_point is not None:
                blue_laser_detected = True
                
                # 绘制蓝紫色激光点（蓝色）
                cv2.circle(result_image, blue_laser_point, 6, (255, 0, 0), 3)
                cv2.circle(result_image, blue_laser_point, 2, (255, 0, 0), -1)
                cv2.putText(result_image, f"蓝紫激光: ({blue_laser_point[0]}, {blue_laser_point[1]})", 
                           (blue_laser_point[0]+15, blue_laser_point[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 打印蓝紫色激光信息
                print("蓝紫色激光检测信息:")
                print(f"蓝紫色激光坐标: {blue_laser_point}")
                
                # 如果同时检测到靶心和激光，计算误差
                if target_center is not None:
                    err_x = blue_laser_point[0] - target_center[0]
                    err_y = blue_laser_point[1] - target_center[1]
                    
                    # 画连接线
                    cv2.line(result_image, target_center, blue_laser_point, (255, 255, 0), 2)
                    
                    # 显示误差信息
                    cv2.putText(result_image, f"Error: ({err_x}, {err_y})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    print(f"激光与靶心误差: X={err_x}, Y={err_y}")
                    
                    # 发布误差数据
                    error_msg = String()
                    error_msg.data = f"blue_laser,{err_x},{err_y}"
                    self.target_publisher.publish(error_msg)
                    self.get_logger().info(f"发布蓝紫激光误差: {error_msg.data}")
            
            # 发布目标数据
            if target_message:
                msg_pub = String()
                msg_pub.data = target_message
                self.target_publisher.publish(msg_pub)
                self.get_logger().info(f"发布目标数据: {target_message}")
            
            # 添加帧率显示
            cv2.putText(result_image, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 添加图像尺寸信息
            cv2.putText(result_image, f"Size: {cv_image.shape[1]}x{cv_image.shape[0]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加检测状态显示（右上角）
            status_text = ""
            status_color = (0, 0, 255)  # 默认红色
            
            if rect_detected and circle_detected and blue_laser_detected:
                status_text = "All Detected"
                status_color = (0, 255, 0)  # 绿色
            elif rect_detected and circle_detected:
                status_text = "Target Detected"
                status_color = (0, 255, 255)  # 黄色
            elif rect_detected:
                status_text = "Rectangle Detected"
                status_color = (0, 165, 255)  # 橙色
            else:
                status_text = "No Target"
                status_color = (0, 0, 255)  # 红色
            
            # 计算文字位置（右上角）
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = result_image.shape[1] - text_size[0] - 10
            text_y = 30
            
            # 添加黑色背景确保文字可读性
            cv2.rectangle(result_image, (text_x-5, text_y-25), (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
            cv2.putText(result_image, status_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 只显示最终结果窗口
            cv2.imshow('Target Detection Result', result_image)
            cv2.waitKey(1)
            
            total_end_time = time.time()
            processing_time = (total_end_time - total_start_time) * 1000
            
            # 打印处理结果总结
            print(f"处理耗时: {processing_time:.2f}ms")
            print(f"检测结果 - 矩形: {'✓' if rect_detected else '✗'}, 圆形: {'✓' if circle_detected else '✗'}, 蓝紫激光: {'✓' if blue_laser_detected else '✗'}")
            print("=" * 50)
            
            self.get_logger().info(f"总处理耗时: {processing_time:.2f}ms, "
                                 f"矩形检测: {'成功' if rect_detected else '失败'}, "
                                 f"圆形检测: {'成功' if circle_detected else '失败'}, "
                                 f"蓝紫激光: {'成功' if blue_laser_detected else '失败'}")
        
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")
    
    def __del__(self):
        """析构函数，释放摄像头资源"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def detect_nested_rectangles(self, img_raw):
        """检测嵌套矩形 - 优化版本"""
        start_time = time.time()
        
        if img_raw is None:
            return None, None, None, None
        
        # 图像预处理
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_bilateral = cv2.bilateralFilter(img_gray, 9, 10, 10)
        kernel = np.ones((5, 5), np.uint8)
        img_close = cv2.morphologyEx(img_bilateral, cv2.MORPH_CLOSE, kernel)
        edged = cv2.Canny(img_close, 50, 150)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选矩形
        rectangles = []
        for i, contour in enumerate(contours):
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(approx)
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
        outer_rect = None
        inner_rect = None
        
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
        
        end_time = time.time()
        self.get_logger().debug(f"矩形检测耗时: {(end_time - start_time)*1000:.2f}ms")
        
        return outer_rect, inner_rect, img_raw, None
    
    def detect_circles(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[int, int, int]], Optional[str], np.ndarray]:
        """圆形检测主函数 - 返回靶心坐标和目标圆信息"""
        start_time = time.time()
        result_frame = frame.copy()

        # ------------------ 1. 灰度化 ------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------ 2. 双边滤波 ------------------
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        # ------------------ 3. ROI提取 ------------------
        height, width = bilateral.shape[:2]
        margin = 100
        x1, y1 = max(0, margin), max(0, margin)
        x2, y2 = min(width - margin, width), min(height - margin, height)

        if x2 <= x1 or y2 <= y1:
            roi = bilateral
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            roi = bilateral[y1:y2, x1:x2]

        # ------------------ 4. 自适应阈值处理 ------------------
        roi_thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 将ROI结果映射回全图
        thresh_full = np.zeros_like(bilateral)
        thresh_full[y1:y2, x1:x2] = roi_thresh

        # ------------------ 5. 形态学操作 ------------------
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_rect = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(thresh_full, cv2.MORPH_OPEN, kernel_rect)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_ellipse)

        # ------------------ 6. 轮廓检测 ------------------
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
            end_time = time.time()
            self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
            return result_frame, None, None, None, None

        hierarchy = hierarchy[0]

        # ------------------ 7. 筛选候选圆 ------------------
        candidate_circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 200: 
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # 添加最小半径约束条件
            if radius < 30:
                continue
            candidate_circles.append([int(x), int(y), int(radius), int(area)])

        if not candidate_circles:
            end_time = time.time()
            self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
            return result_frame, None, None, None, None

        candidate_circles = np.array(candidate_circles)

        # ------------------ 8. 找到最小面积圆作为靶心 ------------------
        min_area_idx = np.argmin(candidate_circles[:, 3])
        innerest_circle = candidate_circles[min_area_idx]
        innerest_x, innerest_y, innerest_r, innerest_area = innerest_circle
        
        # 再次检查最小圆的半径是否满足条件
        if innerest_r < 30:
            end_time = time.time()
            self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms - 最小圆半径不满足条件")
            return result_frame, None, None, None, None
            
        target_center = (int(innerest_x), int(innerest_y))
        cv2.circle(result_frame, target_center, innerest_r, (0, 0, 255), 2)
        cv2.circle(result_frame, target_center, 3, (0, 0, 255), -1)
        cv2.putText(result_frame, f"靶心: {target_center}", 
                    (target_center[0] + 10, target_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ------------------ 9. 筛选目标圆 ------------------
        target_circles = []
        for circle in candidate_circles:
            x, y, r, area = circle
            if 90 <= r <= 120:
                target_circles.append([x, y, r])
                cv2.circle(result_frame, (int(x), int(y)), int(r), (255, 0, 0), 2)
                cv2.circle(result_frame, (int(x), int(y)), 3, (255, 0, 0), -1)

        # ------------------ 10. 最终目标圆 ------------------
        target_circle = None
        target_message = f"p,{target_center[0]},{target_center[1]}"

        if len(target_circles) == 1:
            target_circle = tuple(target_circles[0])
        elif len(target_circles) > 1:
            target_circles = np.array(target_circles)
            avg_x = int(np.mean(target_circles[:, 0]))
            avg_y = int(np.mean(target_circles[:, 1]))
            avg_r = int(np.mean(target_circles[:, 2]))
            target_circle = (avg_x, avg_y, avg_r)

        if target_circle is not None:
            tx, ty, tr = target_circle
            cv2.circle(result_frame, (tx, ty), tr, (0, 255, 0), 3)
            cv2.circle(result_frame, (tx, ty), 5, (0, 255, 0), -1)
            cv2.putText(result_frame, f"目标圆: ({tx}, {ty}), R={tr}", 
                        (tx - 80, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            target_message = f"c,{tx},{ty},{tr}"

        end_time = time.time()
        self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
        return result_frame, target_center, target_circle, target_message, None

    def image_callback(self, msg):
        """图像回调函数"""
        total_start_time = time.time()
        
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 从原图像中心裁剪640x480区域
            cv_image = self.crop_center_image(cv_image, 640, 480)
            
            # 计算帧率
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self.prev_time)) if self.prev_time else 0
            self.prev_time = now
            
            # 1. 检测嵌套矩形（直接在640x480图像上检测）
            outer_rect, inner_rect, _, _ = self.detect_nested_rectangles(cv_image)
            
            result_image = cv_image.copy()
            rect_detected = False
            circle_detected = False
            blue_laser_detected = False
            target_message = None
            target_center = None
            target_circle = None
            
            # 画出检测到的矩形并打印信息
            if outer_rect is not None and inner_rect is not None:
                rect_detected = True
                
                # 绘制矩形框
                cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 3)
                cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 3)
                cv2.circle(result_image, outer_rect['center'], 5, (0, 0, 255), -1)
                cv2.circle(result_image, inner_rect['center'], 5, (0, 0, 255), -1)
                
                # 添加矩形标签
                cv2.putText(result_image, "OUTER RECT", 
                           (outer_rect['center'][0]-50, outer_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result_image, "INNER RECT", 
                           (inner_rect['center'][0]-50, inner_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 打印矩形详细信息
                print("=" * 50)
                print("矩形检测信息:")
                print(f"外矩形 - 角点: {outer_rect['corners'].tolist()}")
                print(f"外矩形 - bbox(x,y,w,h): {outer_rect['bbox']}")
                print(f"外矩形 - 中心点: {outer_rect['center']}")
                print(f"外矩形 - 面积: {outer_rect['area']}")
                print(f"内矩形 - 角点: {inner_rect['corners'].tolist()}")
                print(f"内矩形 - bbox(x,y,w,h): {inner_rect['bbox']}")
                print(f"内矩形 - 中心点: {inner_rect['center']}")
                print(f"内矩形 - 面积: {inner_rect['area']}")
            
            # 2. 检测圆形目标（直接在640x480图像上检测）
            circle_result, target_center, target_circle, target_message, _ = self.detect_circles(cv_image)
            
            if target_center is not None or target_circle is not None:
                circle_detected = True
                
                # 绘制靶心
                if target_center is not None:
                    cv2.circle(result_image, target_center, 8, (0, 0, 255), 3)
                    cv2.circle(result_image, target_center, 3, (0, 0, 255), -1)
                    cv2.putText(result_image, f"靶心: ({target_center[0]}, {target_center[1]})", 
                               (target_center[0]+15, target_center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 打印靶心信息
                    print("靶心检测信息:")
                    print(f"靶心坐标: {target_center}")
                
                # 绘制目标圆
                if target_circle is not None:
                    tc_x, tc_y, tc_r = target_circle
                    cv2.circle(result_image, (tc_x, tc_y), tc_r, (0, 255, 0), 3)
                    cv2.circle(result_image, (tc_x, tc_y), 5, (0, 255, 0), -1)
                    cv2.putText(result_image, f"目标圆: ({tc_x}, {tc_y}), R={tc_r}", 
                               (tc_x-100, tc_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 打印目标圆信息
                    print("目标圆检测信息:")
                    print(f"目标圆圆心坐标: ({tc_x}, {tc_y})")
                    print(f"目标圆半径: {tc_r}")
            
            # 3. 检测蓝紫色激光（直接在640x480图像上检测）
            blue_laser_point, _ = self.detect_blue_purple_laser(cv_image, target_center)
            if blue_laser_point is not None:
                blue_laser_detected = True
                
                # 绘制蓝紫色激光点（蓝色）
                cv2.circle(result_image, blue_laser_point, 6, (255, 0, 0), 3)
                cv2.circle(result_image, blue_laser_point, 2, (255, 0, 0), -1)
                cv2.putText(result_image, f"蓝紫激光: ({blue_laser_point[0]}, {blue_laser_point[1]})", 
                           (blue_laser_point[0]+15, blue_laser_point[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 打印蓝紫色激光信息
                print("蓝紫色激光检测信息:")
                print(f"蓝紫色激光坐标: {blue_laser_point}")
                
                # 如果同时检测到靶心和激光，计算误差
                if target_center is not None:
                    err_x = blue_laser_point[0] - target_center[0]
                    err_y = blue_laser_point[1] - target_center[1]
                    
                    # 画连接线
                    cv2.line(result_image, target_center, blue_laser_point, (255, 255, 0), 2)
                    
                    # 显示误差信息
                    cv2.putText(result_image, f"Error: ({err_x}, {err_y})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    print(f"激光与靶心误差: X={err_x}, Y={err_y}")
                    
                    # 发布误差数据
                    error_msg = String()
                    error_msg.data = f"blue_laser,{err_x},{err_y}"
                    self.target_publisher.publish(error_msg)
                    self.get_logger().info(f"发布蓝紫激光误差: {error_msg.data}")
            
            # 发布目标数据
            if target_message:
                msg_pub = String()
                msg_pub.data = target_message
                self.target_publisher.publish(msg_pub)
                self.get_logger().info(f"发布目标数据: {target_message}")
            
            # 添加帧率显示
            cv2.putText(result_image, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 添加图像尺寸信息
            cv2.putText(result_image, f"Size: {cv_image.shape[1]}x{cv_image.shape[0]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加检测状态显示（右上角）
            status_text = ""
            status_color = (0, 0, 255)  # 默认红色
            
            if rect_detected and circle_detected and blue_laser_detected:
                status_text = "All Detected"
                status_color = (0, 255, 0)  # 绿色
            elif rect_detected and circle_detected:
                status_text = "Target Detected"
                status_color = (0, 255, 255)  # 黄色
            elif rect_detected:
                status_text = "Rectangle Detected"
                status_color = (0, 165, 255)  # 橙色
            else:
                status_text = "No Target"
                status_color = (0, 0, 255)  # 红色
            
            # 计算文字位置（右上角）
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = result_image.shape[1] - text_size[0] - 10
            text_y = 30
            
            # 添加黑色背景确保文字可读性
            cv2.rectangle(result_image, (text_x-5, text_y-25), (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
            cv2.putText(result_image, status_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 只显示最终结果窗口
            cv2.imshow('Target Detection Result', result_image)
            cv2.waitKey(1)
            
            total_end_time = time.time()
            processing_time = (total_end_time - total_start_time) * 1000
            
            # 打印处理结果总结
            print(f"处理耗时: {processing_time:.2f}ms")
            print(f"检测结果 - 矩形: {'✓' if rect_detected else '✗'}, 圆形: {'✓' if circle_detected else '✗'}, 蓝紫激光: {'✓' if blue_laser_detected else '✗'}")
            print("=" * 50)
            
            self.get_logger().info(f"总处理耗时: {processing_time:.2f}ms, "
                                 f"矩形检测: {'成功' if rect_detected else '失败'}, "
                                 f"圆形检测: {'成功' if circle_detected else '失败'}, "
                                 f"蓝紫激光: {'成功' if blue_laser_detected else '失败'}")
        
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = TargetDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()