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

# 导入工具函数
from .image_processing_utils import crop_center_and_flip, preprocess_image
from .rectangle_detection import detect_nested_rectangles_optimized
from .geometry_utils import sort_corners
from .visualization_utils import render_results
from .tcp_sender import ImageSender

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

        # tcp发送图片
        self.image_sender = ImageSender('192.168.31.89', 5000)
        self.image_sender.connect()
        
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
        self.cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
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
                cropped_frame = crop_center_and_flip(frame, 960, 720)
                
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
    
    def compute_perspective_transform(self, inner_rect):
        """计算从inner_rect角点到640x480的透视变换矩阵（带缓存优化）"""
        if inner_rect is None:
            return None, None
            
        # 获取inner_rect的四个角点，按顺序排列
        corners = inner_rect['corners']
        
        # 对角点进行排序：左上、右上、右下、左下
        self.corners = sort_corners(corners)
        
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
        """基于透视变换和先验知识直接确定靶心位置和目标圆"""
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
            processed_data = preprocess_image(cv_image, self.timing_stats)
            
            # 2. 矩形检测
            outer_rect, inner_rect = detect_nested_rectangles_optimized(processed_data['edged'], self.timing_stats)
            
            # 3. 根据透视变换和先验知识获取靶心位置和目标圆
            target_center, target_circle, perspective_vis = self.get_target_from_perspective(
                cv_image, inner_rect
            )
            
            # 4. 渲染和发布
            render_start = time.time()
            result_image = render_results(
                cv_image, outer_rect, inner_rect, 
                target_center, target_circle, self.fps, self.frame_count, self.target_circle_area
            )
            render_time = (time.time() - render_start) * 1000
            self.timing_stats['rendering'].append(render_time)
            
            # 5. 发布数据
            self.publish_detection_data(target_center, target_circle)
            
            # 6. 发布透视变换矩阵（如果开启）
            self.publish_perspective_matrix()
            
            # 7. 显示结果
            # cv2.imshow('Target Detection Result', result_image)
            
            self.image_sender.send_image(result_image)  
            
            # 8. 性能统计
            total_time = (time.time() - total_start_time) * 1000
            self.timing_stats['total'].append(total_time)
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            import traceback
            traceback.print_exc()

    def publish_detection_data(self, target_center, target_circle):
        """发布检测数据"""
        # 只有当publish_target_data为True时才发布数据
        if not self.publish_target_data:
            return
            
        if target_center:
            if self.target_circle_area <= 9000: # 向上偏移15
                target_message = f"p,{target_center[0]},{target_center[1]-10}"
            elif self.target_circle_area >= 20000: # 向下偏移15
                target_message = f"p,{target_center[0]},{target_center[1]-10}"
            else:
                target_message = f"p,{target_center[0]},{target_center[1]-10}-"
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
        if hasattr(self, 'image_sender'):
            self.image_sender.disconnect()
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
