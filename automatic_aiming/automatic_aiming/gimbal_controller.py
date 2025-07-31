#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import re

class GimbalController(Node):
    """云台控制器节点"""
    
    def __init__(self):
        super().__init__('gimbal_controller')
        
        # 图像中心配置 - 可灵活修改
        self.image_width = 640
        self.image_height = 480
        self.image_center_x = self.image_width // 2   # 320
        self.image_center_y = self.image_height // 2  # 240
        
        # 创建订阅者 - 订阅目标数据
        self.target_subscriber = self.create_subscription(
            String,
            '/target_data',
            self.target_callback,
            10
        )
        
        # 创建发布者 - 发布云台控制指令
        self.gimbal_publisher = self.create_publisher(
            String,
            '/uart1_sender_topic',
            10
        )
        
        # 误差统计
        self.error_count = 0
        self.last_target_center = None
        
        self.get_logger().info(f'云台控制器节点已启动')
        self.get_logger().info(f'图像中心设置为: ({self.image_center_x}, {self.image_center_y})')
        self.get_logger().info(f'订阅话题: /target_data')
        self.get_logger().info(f'发布话题: /uart1_sender_topic')
    
    def parse_target_data(self, data_str):
        """
        解析目标数据
        输入格式: "p,272,252" 或 "c,272,252,50"
        返回: (x, y) 或 None
        """
        try:
            # 移除可能的空白字符
            data_str = data_str.strip()
            
            # 使用正则表达式解析数据
            # 匹配 p,x,y 或 c,x,y,r 格式
            pattern = r'^([pc]),(\d+),(\d+)(?:,(\d+))?$'
            match = re.match(pattern, data_str)
            
            if match:
                data_type = match.group(1)  # 'p' 或 'c'
                x = int(match.group(2))
                y = int(match.group(3))
                
                self.get_logger().debug(f'解析目标数据成功: 类型={data_type}, 坐标=({x}, {y})')
                return (x, y)
            else:
                self.get_logger().warning(f'目标数据格式不正确: {data_str}')
                return None
                
        except Exception as e:
            self.get_logger().error(f'解析目标数据时出错: {data_str}, 错误: {str(e)}')
            return None
    
    def calculate_error(self, target_center):
        """
        计算目标中心与图像中心的误差
        参数: target_center (x, y)
        返回: (x_error, y_error)
        """
        if target_center is None:
            return None, None
        
        target_x, target_y = target_center
        
        # 计算误差 (目标位置 - 图像中心)
        x_error = target_x - self.image_center_x
        y_error = target_y - self.image_center_y
        
        return x_error, y_error
    
    def format_gimbal_command(self, x_error, y_error):
        """
        格式化云台控制指令
        格式: "@0,x偏差,y偏差\r\n"
        """
        # 格式化指令字符串
        command = f"@0,{x_error},{y_error}\r\n"
        return command
    
    def target_callback(self, msg):
        """目标数据回调函数"""
        try:
            # 解析目标数据
            target_center = self.parse_target_data(msg.data)
            
            if target_center is None:
                self.error_count += 1
                self.get_logger().warning(f'无法解析目标数据: {msg.data}')
                return
            
            # 计算误差
            x_error, y_error = self.calculate_error(target_center)
            
            if x_error is None or y_error is None:
                self.get_logger().error('计算误差失败')
                return
            
            # 格式化并发布云台控制指令
            gimbal_command = self.format_gimbal_command(x_error, y_error)
            
            # 发布指令
            cmd_msg = String()
            cmd_msg.data = gimbal_command
            self.gimbal_publisher.publish(cmd_msg)
            
            # 记录信息
            self.last_target_center = target_center
            target_x, target_y = target_center
            
            self.get_logger().info(
                f'目标: ({target_x}, {target_y}) | '
                f'中心: ({self.image_center_x}, {self.image_center_y}) | '
                f'误差: ({x_error:+d}, {y_error:+d}) | '
                f'指令: {gimbal_command}'
            )
            
        except Exception as e:
            self.error_count += 1
            self.get_logger().error(f'处理目标数据时出错: {str(e)}')
    
    def update_image_center(self, width, height):
        """动态更新图像中心配置"""
        self.image_width = width
        self.image_height = height
        self.image_center_x = width // 2
        self.image_center_y = height // 2
        
        self.get_logger().info(f'图像中心已更新为: ({self.image_center_x}, {self.image_center_y})')
    
    def get_status(self):
        """获取节点状态信息"""
        status = {
            'image_center': (self.image_center_x, self.image_center_y),
            'image_size': (self.image_width, self.image_height),
            'last_target': self.last_target_center,
            'error_count': self.error_count
        }
        return status

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        # 创建云台控制器节点
        gimbal_controller = GimbalController()
        
        # 运行节点
        rclpy.spin(gimbal_controller)
        
    except KeyboardInterrupt:
        print('\n云台控制器节点被用户中断')
    except Exception as e:
        print(f'云台控制器节点运行出错: {str(e)}')
    finally:
        # 清理资源
        if 'gimbal_controller' in locals():
            gimbal_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
