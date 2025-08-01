#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import time

class WarpDataPublisher(Node):
    def __init__(self):
        super().__init__('warp_data_publisher_test')
        
        # 发布话题
        self.publisher = self.create_publisher(String, '/warp_data', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)  # 每2秒发布一次
        
        self.get_logger().info('测试节点已启动，开始随机发布warp_data数据')

    def timer_callback(self):
        """
        定时器回调：生成随机透视变换矩阵H和圆参数并发布
        """
        # 生成一个简单的随机透视矩阵 (模拟远近角度变化)
        H = np.array([
            [1.0 + np.random.uniform(-0.05, 0.05), np.random.uniform(-0.02, 0.02), np.random.uniform(-50, 50)],
            [np.random.uniform(-0.02, 0.02), 1.0 + np.random.uniform(-0.05, 0.05), np.random.uniform(-30, 30)],
            [np.random.uniform(-1e-4, 1e-4), np.random.uniform(-1e-4, 1e-4), 1.0]
        ])
        
        # 生成随机圆参数 (模拟检测结果)
        cx = 300 + np.random.uniform(-10, 10)
        cy = 250 + np.random.uniform(-10, 10)
        radius = 100 + np.random.uniform(-5, 5)
        
        # 构造消息字符串
        H_flat = ','.join([f"{v:.6f}" for v in H.flatten()])
        circle_str = f"{cx:.3f},{cy:.3f},{radius:.3f}"
        
        msg = String()
        msg.data = f"M:{H_flat}\nC:{circle_str}"
        self.publisher.publish(msg)
        
        self.get_logger().info(f"发布数据 -> 矩阵H(3x3)和圆(cx={cx:.1f}, cy={cy:.1f}, r={radius:.1f})")

def main(args=None):
    rclpy.init(args=args)
    node = WarpDataPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
