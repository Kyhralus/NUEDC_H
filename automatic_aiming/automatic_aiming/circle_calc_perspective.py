#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import threading
import time

class CircleMappingCache:
    def __init__(self, max_size=50, tol=1e-4):
        self.cache = {}
        self.max_size = max_size
        self.tol = tol

    def _make_key(self, cx, cy, r, M):
        return (
            round(cx, 3), round(cy, 3), round(r, 3),
            tuple(np.round(M.flatten(), 4))
        )

    def get(self, cx, cy, r, M):
        return self.cache.get(self._make_key(cx, cy, r, M), None)

    def add(self, cx, cy, r, M, mapped_points):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[self._make_key(cx, cy, r, M)] = mapped_points

def map_circle_points_to_original(cx_trans, cy_trans, r_trans, H, num_points=300):
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    circle_points = np.stack([
        cx_trans + r_trans * np.cos(theta),
        cy_trans + r_trans * np.sin(theta),
        np.ones(num_points)
    ], axis=1)
    H_inv = np.linalg.inv(H)
    mapped = (circle_points @ H_inv.T)
    return mapped[:, :2] / mapped[:, 2:3]

def cached_circle_mapping(cx_trans, cy_trans, r_trans, H, cache=None, num_points=300):
    if cache is None:
        cache = CircleMappingCache()
    result = cache.get(cx_trans, cy_trans, r_trans, H)
    if result is not None:
        return result, cache
    mapped_points = map_circle_points_to_original(cx_trans, cy_trans, r_trans, H, num_points)
    cache.add(cx_trans, cy_trans, r_trans, H, mapped_points)
    return mapped_points, cache

class CirclePointCalculatorPerspective(Node):
    def __init__(self):
        super().__init__('circle_point_calculator')
        
        self.subscription = self.create_subscription(
            String,
            '/warp_data',
            self.warp_data_callback,
            10
        )
        self.publisher = self.create_publisher(String, '/draw_circle_date', 10)
        self.cache = CircleMappingCache()
        
        # 采样和发布配置
        self.num_points = 600           # 圆周上的采样点总数
        self.publish_rate = 30.0        # Hz，发布频率
        self.global_point_index = 0     # 全局计数器，指示当前发布点的序号
        self.thread_lock = threading.Lock()  # 线程锁

        # 当前数据
        self.current_matrix = None      # 当前透视变换矩阵
        self.current_circle = None      # 当前圆参数
        self.circle_points = None       # 当前圆的采样点

        # 创建30Hz定时器，用于按频率发布点
        self.publish_timer = self.create_timer(1.0/self.publish_rate, self.publish_next_point)

        # 添加状态标志
        self.has_received_data = False
        self.last_point_time = None
        self.data_updated = False  # 标记数据是否已更新
        self.last_update_time = 0.0  # 记录上次数据更新时间

        self.get_logger().info('透视变换圆形采样点计算节点已启动')
        self.get_logger().info(f'采样点数量: {self.num_points}')
        self.get_logger().info(f'发布频率: {self.publish_rate}Hz')
        self.get_logger().info(f'订阅话题: /warp_data')
        self.get_logger().info(f'发布话题: /draw_circle_date')

    def parse_warp_data(self, data_str):
        try:
            lines = data_str.strip().split('\n')
            matrix = None
            circle = None
            for line in lines:
                if line.startswith('M:'):
                    vals = [float(x) for x in line[2:].split(',')]
                    if len(vals) == 9:
                        matrix = np.array(vals).reshape(3, 3)
                elif line.startswith('C:'):
                    vals = [float(x) for x in line[2:].split(',')]
                    if len(vals) == 3:
                        cx, cy, radius = vals
                        if radius > 0:
                            circle = (cx, cy, radius)
            return matrix, circle
        except Exception as e:
            self.get_logger().error(f'解析数据失败: {e}')
            return None, None

    def warp_data_callback(self, msg):
        """接收透视变换矩阵和圆参数，重新计算采样点"""
        matrix, circle = self.parse_warp_data(msg.data)
        if matrix is not None and circle is not None:
            with self.thread_lock:
                self.current_matrix = matrix
                self.current_circle = circle
                # 重新计算圆上的采样点（不重置发布序号）
                self.circle_points = self.generate_circle_points(circle, self.num_points)
                
                # 标记数据已更新
                self.data_updated = True
                self.last_update_time = time.time()
                
                # 标记已收到数据
                if not self.has_received_data:
                    self.has_received_data = True
                    self.get_logger().info(f'首次接收到数据，开始持续发布点')
                else:
                    self.get_logger().info(f'更新采样点数据 - 圆心:({circle[0]:.1f},{circle[1]:.1f}), 半径:{circle[2]:.1f}')

    def generate_circle_points(self, circle, num_points):
        """生成圆上采样点，逆时针，最上方为起点"""
        cx, cy, r = circle
        # 生成逆时针顺序的角度，以最上方为起点（-π/2对应最上方）
        theta = np.linspace(-np.pi/2, 3*np.pi/2, num_points, endpoint=False)
        # 计算圆周上的点坐标
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        points = np.column_stack((x, y))
        return points

    def publish_next_point(self):
        """定时器回调，发布下一个采样点"""
        with self.thread_lock:
            # 如果尚未接收到任何数据，则跳过发布
            if self.circle_points is None or self.current_matrix is None:
                return
            
            # 每秒打印一次状态信息，显示是否使用保持的数据
            now = time.time()
            if now - self.last_update_time > 1.0 and not self.data_updated:
                # 每秒记录一次使用保持数据的信息
                if int(now) != int(self.last_update_time):
                    if self.current_circle:
                        self.get_logger().info(f'[keep] 使用原有采样点数据 - 圆心:({self.current_circle[0]:.1f},{self.current_circle[1]:.1f}), 半径:{self.current_circle[2]:.1f}')
            
            # 重置数据更新标志
            self.data_updated = False
            
            # 计算当前发布频率
            if self.last_point_time is not None:
                actual_interval = now - self.last_point_time
                actual_freq = 1.0 / actual_interval if actual_interval > 0 else 0
                # 每100个点记录一次发布频率
                if self.global_point_index % 100 == 0:
                    self.get_logger().info(f'点发布频率: {actual_freq:.1f}Hz')
            self.last_point_time = now
            
            # 取当前索引对应的点
            idx = self.global_point_index % self.num_points
            pt_trans = self.circle_points[idx]
            
            # 逆透视变换到原图像坐标系
            pt_homo = np.array([pt_trans[0], pt_trans[1], 1.0])
            H_inv = np.linalg.inv(self.current_matrix)
            mapped = H_inv @ pt_homo
            mapped_xy = mapped[:2] / mapped[2]
            
            # 发布转换后的点（向下取整）
            point_msg = String()
            point_msg.data = f"p,{int(mapped_xy[0])},{int(mapped_xy[1])}"
            self.publisher.publish(point_msg)
            # 更新全局计数器
            self.global_point_index = (self.global_point_index + 1) % self.num_points

def main(args=None):
    rclpy.init(args=args)
    
    node = CirclePointCalculatorPerspective()
    
    try:
        print("节点启动中... 按 Ctrl+C 退出")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n正在关闭节点...")
    except Exception as e:
        print(f"节点运行错误: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("节点已安全关闭")

if __name__ == '__main__':
    main()
