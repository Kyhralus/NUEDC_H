# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from collections import deque

# ROS2 相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class EnhancedCircleDetector:
    """增强版圆形检测器类"""
    def __init__(self):
        # 稳定性参数
        self.history_size = 7
        self.center_history = deque(maxlen=self.history_size)
        self.quality_history = deque(maxlen=self.history_size)
        
        # 检测参数
        self.min_area = 50
        self.max_area = 10000
        self.min_circularity = 0.65
        self.duplicate_dist_thresh = 10
        self.duplicate_radius_thresh = 10
        
        # 质量评估参数
        self.min_quality_score = 0.5
        self.quality_weight_circularity = 0.4
        self.quality_weight_area_ratio = 0.3
        self.quality_weight_stability = 0.3

class TargetDetectionNode(Node):
    """目标检测ROS2节点"""
    
    def __init__(self):
        super().__init__('target_detection_node')
        
        # 初始化检测器
        self.detector = EnhancedCircleDetector()
        self.bridge = CvBridge()
        
        # 创建订阅者和发布者
        self.image_subscriber = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.target_publisher = self.create_publisher(
            String,
            '/target_data',
            10
        )
        
        # 帧率计算相关
        self.prev_time = time.time()
        self.fps = 0
        
        self.get_logger().info('目标检测节点已启动')
    
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
        
        # 创建矩形检测结果图像用于显示
        rect_debug_img = img_raw.copy()
        
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
                # 在调试图像上画出所有候选矩形（蓝色）
                cv2.drawContours(rect_debug_img, [approx], 0, (255, 0, 0), 2)
                cv2.putText(rect_debug_img, f"ID:{i} Area:{int(area)}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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
                        # 在调试图像上高亮显示嵌套矩形对
                        cv2.drawContours(rect_debug_img, [outer['approx']], 0, (0, 255, 0), 3)
                        cv2.drawContours(rect_debug_img, [inner['approx']], 0, (0, 0, 255), 3)
                        cv2.putText(rect_debug_img, "OUTER", 
                                   (outer['center'][0]-30, outer['center'][1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(rect_debug_img, "INNER", 
                                   (inner['center'][0]-30, inner['center'][1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        break
            if outer_rect is not None:
                break
        
        # 添加检测信息到调试图像
        info_text = f"Rectangles Found: {len(rectangles)}"
        cv2.putText(rect_debug_img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        nested_status = "Nested Pair: Found" if (outer_rect and inner_rect) else "Nested Pair: Not Found"
        cv2.putText(rect_debug_img, nested_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        end_time = time.time()
        self.get_logger().debug(f"矩形检测耗时: {(end_time - start_time)*1000:.2f}ms")
        
        return outer_rect, inner_rect, img_raw, rect_debug_img
    
    def detect_circles(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[int, int, int]], np.ndarray]:
        """圆形检测主函数 - 只返回检测结果，不组织消息"""
        start_time = time.time()
        result_frame = frame.copy()
        
        # 创建圆形检测调试图像
        circle_debug_img = frame.copy()
        
        # 图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # ROI处理
        height, width = bilateral.shape[:2]
        margin = 90
        x1, y1 = max(0, margin), max(0, margin)
        x2, y2 = min(width - margin, width), min(height - margin, height)
        
        if x2 <= x1 or y2 <= y1:
            roi = bilateral
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            roi = bilateral[y1:y2, x1:x2]
        
        # 自适应阈值处理
        roi_thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        thresh_full = np.zeros_like(bilateral)
        thresh_full[y1:y2, x1:x2] = roi_thresh
        
        # 形态学操作
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_rect = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(thresh_full, cv2.MORPH_OPEN, kernel_rect)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_ellipse)
        
        # 轮廓检测
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
            # 添加调试信息
            cv2.putText(circle_debug_img, "No Contours Found", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            end_time = time.time()
            self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
            return result_frame, None, None, circle_debug_img
        
        hierarchy = hierarchy[0]
        
        # 候选圆筛选
        candidate_circles = []
        all_contours_count = len(contours)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 200:  # 面积筛选
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.8:  # 圆度筛选
                continue
                
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            candidate_circles.append([int(x), int(y), int(radius), int(area)])
            
            # 在调试图像上画出所有候选圆（黄色）
            cv2.circle(circle_debug_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(circle_debug_img, f"C{i}:R{int(radius)}", 
                       (int(x)-20, int(y)-int(radius)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 添加检测统计信息
        cv2.putText(circle_debug_img, f"Total Contours: {all_contours_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(circle_debug_img, f"Candidate Circles: {len(candidate_circles)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if not candidate_circles:
            cv2.putText(circle_debug_img, "No Valid Circles Found", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            end_time = time.time()
            self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
            return result_frame, None, None, circle_debug_img
        
        candidate_circles = np.array(candidate_circles)
        
        # 找到面积最小的圆作为靶心
        min_area_idx = np.argmin(candidate_circles[:, 3])
        innerest_circle = candidate_circles[min_area_idx]
        innerest_x, innerest_y, innerest_r, innerest_area = innerest_circle
        
        # 定义靶心坐标
        target_center = (int(innerest_x), int(innerest_y))
        
        # 画出靶心（红色）
        cv2.circle(result_frame, target_center, innerest_r, (0, 0, 255), 2)
        cv2.circle(result_frame, target_center, 3, (0, 0, 255), -1)
        cv2.putText(result_frame, f"靶心: {target_center}", 
                    (target_center[0] + 10, target_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 在调试图像上高亮靶心
        cv2.circle(circle_debug_img, target_center, innerest_r, (0, 0, 255), 3)
        cv2.putText(circle_debug_img, "BULLSEYE", (target_center[0]-30, target_center[1]+innerest_r+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 筛选目标圆（半径在90-120之间）
        target_circles = []
        for circle in candidate_circles:
            x, y, r, area = circle
            if 90 <= r <= 120:
                target_circles.append([x, y, r])
                # 画出候选目标圆（蓝色）
                cv2.circle(result_frame, (int(x), int(y)), int(r), (255, 0, 0), 2)
                cv2.circle(result_frame, (int(x), int(y)), 3, (255, 0, 0), -1)
                # 在调试图像上标记目标圆
                cv2.circle(circle_debug_img, (int(x), int(y)), int(r), (255, 0, 0), 3)
                cv2.putText(circle_debug_img, "TARGET", (int(x)-30, int(y)-int(r)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(circle_debug_img, f"Target Circles (R:90-120): {len(target_circles)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 计算最终目标圆
        target_circle = None
        
        if len(target_circles) == 1:
            target_circle = (int(target_circles[0][0]), int(target_circles[0][1]), int(target_circles[0][2]))
        elif len(target_circles) > 1:
            target_circles = np.array(target_circles)
            avg_x = int(np.mean(target_circles[:, 0]))
            avg_y = int(np.mean(target_circles[:, 1]))
            avg_r = int(np.mean(target_circles[:, 2]))
            target_circle = (avg_x, avg_y, avg_r)
        
        # 画出最终目标圆（绿色）
        if target_circle is not None:
            tx, ty, tr = target_circle
            cv2.circle(result_frame, (tx, ty), tr, (0, 255, 0), 3)
            cv2.circle(result_frame, (tx, ty), 5, (0, 255, 0), -1)
            cv2.putText(result_frame, f"目标圆: ({tx}, {ty}), R={tr}", 
                        (tx - 80, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 在调试图像上标记最终目标
            cv2.circle(circle_debug_img, (tx, ty), tr, (0, 255, 0), 4)
            cv2.putText(circle_debug_img, "FINAL TARGET", (tx-50, ty+tr+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(circle_debug_img, f"Final Target Found: ({tx}, {ty})", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(circle_debug_img, f"Using Bullseye Only: {target_center}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        end_time = time.time()
        self.get_logger().debug(f"圆形检测耗时: {(end_time - start_time)*1000:.2f}ms")
        
        # 只返回检测结果，不组织消息
        return result_frame, target_center, target_circle, circle_debug_img

    def image_callback(self, msg):
        """图像回调函数"""
        total_start_time = time.time()
        
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 计算帧率
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self.prev_time)) if self.prev_time else 0
            self.prev_time = now
            
            # 1. 检测嵌套矩形
            outer_rect, inner_rect, raw_img, rect_debug_img = self.detect_nested_rectangles(cv_image)
            
            result_image = cv_image.copy()
            rect_detected = False
            circle_detected = False
            circle_debug_img = None
            
            # 画出检测到的矩形
            if outer_rect is not None and inner_rect is not None:
                rect_detected = True
                cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 3)
                cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 3)
                cv2.circle(result_image, outer_rect['center'], 5, (0, 0, 255), -1)
                cv2.circle(result_image, inner_rect['center'], 5, (0, 0, 255), -1)
                
                # 添加矩形信息
                cv2.putText(result_image, "OUTER RECT", 
                           (outer_rect['center'][0]-50, outer_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result_image, "INNER RECT", 
                           (inner_rect['center'][0]-50, inner_rect['center'][1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 2. 检测圆形目标
            if outer_rect is not None:
                # 提取ROI区域
                corners = outer_rect['corners']
                xs = corners[:,0]
                ys = corners[:,1]
                x_min = max(xs.min() - 20, 0)
                x_max = min(xs.max() + 20, cv_image.shape[1])
                y_min = max(ys.min() - 20, 0)
                y_max = min(ys.max() + 20, cv_image.shape[0])
                
                if y_max > y_min and x_max > x_min:
                    roi = raw_img[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        roi_resized = cv2.resize(roi, (640, 480))
                        
                        # 圆形检测 - 更新函数调用，只返回检测结果
                        circle_result, target_center, target_circle, circle_debug_img = self.detect_circles(roi_resized)
                        
                        # 初始化消息发布列表
                        messages_to_publish = []
                        
                        # 处理靶心检测结果
                        if target_center is not None:
                            circle_detected = True
                            
                            # 映射靶心坐标到原图
                            scale_x = (x_max - x_min) / 640
                            scale_y = (y_max - y_min) / 480
                            mapped_center_x = int(target_center[0] * scale_x + x_min)
                            mapped_center_y = int(target_center[1] * scale_y + y_min)
                            
                            # 在主图像上画出靶心（红色）
                            cv2.circle(result_image, (mapped_center_x, mapped_center_y), 8, (0, 0, 255), 3)
                            cv2.circle(result_image, (mapped_center_x, mapped_center_y), 3, (0, 0, 255), -1)
                            cv2.putText(result_image, f"靶心: ({mapped_center_x}, {mapped_center_y})", 
                                       (mapped_center_x+15, mapped_center_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # 组织靶心消息
                            target_center_msg = f"p,{mapped_center_x},{mapped_center_y}"
                            messages_to_publish.append(target_center_msg)
                            self.get_logger().info(f"靶心检测到: {target_center_msg}")
                        
                        # 处理目标圆检测结果
                        if target_circle is not None:
                            tc_x, tc_y, tc_r = target_circle
                            mapped_circle_x = int(tc_x * scale_x + x_min)
                            mapped_circle_y = int(tc_y * scale_y + y_min)
                            mapped_circle_r = int(tc_r * scale_x)  # 缩放半径
                            
                            # 在主图像上画出目标圆（绿色）
                            cv2.circle(result_image, (mapped_circle_x, mapped_circle_y), mapped_circle_r, (0, 255, 0), 3)
                            cv2.circle(result_image, (mapped_circle_x, mapped_circle_y), 5, (0, 255, 0), -1)
                            cv2.putText(result_image, f"目标圆: ({mapped_circle_x}, {mapped_circle_y}), R={mapped_circle_r}", 
                                       (mapped_circle_x-100, mapped_circle_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # 组织目标圆消息
                            target_circle_msg = f"c,{mapped_circle_x},{mapped_circle_y},{mapped_circle_r}"
                            messages_to_publish.append(target_circle_msg)
                            self.get_logger().info(f"目标圆检测到: {target_circle_msg}")
                        
                        # 发布所有检测到的目标数据
                        for message in messages_to_publish:
                            msg_pub = String()
                            msg_pub.data = message
                            self.target_publisher.publish(msg_pub)
                            self.get_logger().info(f"发布目标数据: {message}")
                            # 添加小延迟避免消息冲突
                            time.sleep(0.01)
            
            # 添加帧率显示
            cv2.putText(result_image, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 添加检测状态显示（右上角）
            status_text = ""
            status_color = (0, 0, 255)  # 默认红色
            
            if rect_detected and circle_detected:
                status_text = "Target Detected"
                status_color = (0, 255, 0)  # 绿色
            elif rect_detected:
                status_text = "Rectangle Detected"
                status_color = (0, 255, 255)  # 黄色
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
            
            # 显示主结果窗口
            cv2.imshow('Target Detection Result', result_image)
            
            # 显示矩形检测调试窗口
            if rect_debug_img is not None:
                cv2.imshow('Rectangle Detection Debug', rect_debug_img)
            
            # 显示圆形检测调试窗口
            if circle_debug_img is not None:
                cv2.imshow('Circle Detection Debug', circle_debug_img)
            
            cv2.waitKey(1)
            
            total_end_time = time.time()
            self.get_logger().info(f"总处理耗时: {(total_end_time - total_start_time)*1000:.2f}ms, "
                                 f"矩形检测: {'成功' if rect_detected else '失败'}, "
                                 f"圆形检测: {'成功' if circle_detected else '失败'}")
        
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