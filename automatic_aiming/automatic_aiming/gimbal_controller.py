#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
import re

class GimbalController(Node):
    """云台控制器节点"""
    
    def __init__(self):
        super().__init__('gimbal_controller')
        
        # 图像中心配置 - 可灵活修改
        self.image_width = 800
        self.image_height = 600
        self.image_center_x = self.image_width // 2   # 400
        self.image_center_y = self.image_height // 2  # 300
        
        # 阈值配置
        self.error_threshold = 5           # 最大允许误差
        self.success_required_count = 10   # 连续成功次数要求
        self.success_counter = 0           # 计数器

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
        
        # 创建客户端
        self.shoot_client = self.create_client(SetBool, '/shoot_status')
        while not self.shoot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待服务端 /shoot_status 启动...")

        # 误差统计
        self.error_count = 0
        self.last_target_center = None
        
        # 存储最新的云台控制指令
        self.latest_gimbal_command = None
        
        # 创建30Hz定时器用于发送云台控制指令
        self.gimbal_timer = self.create_timer(0.03, self.gimbal_timer_callback)

        self.get_logger().info(f'云台控制器节点已启动')
        self.get_logger().info(f'图像中心设置为: ({self.image_center_x}, {self.image_center_y})')
        self.get_logger().info(f'订阅话题: /target_data')
        self.get_logger().info(f'发布话题: /uart1_sender_topic (30Hz)')
        self.get_logger().info(f'控制频率: 30Hz')
    

    def gimbal_timer_callback(self):
        """30Hz定时器回调函数，用于发送云台控制指令"""
        if self.latest_gimbal_command is not None:
            cmd_msg = String()
            cmd_msg.data = self.latest_gimbal_command
            self.gimbal_publisher.publish(cmd_msg)
            self.get_logger().debug(f'发送云台指令: {self.latest_gimbal_command.strip()}')

    def call_shoot_service(self, success: bool):
        """调用服务端，发送打靶成功信号"""
        req = SetBool.Request()
        req.data = success
        future = self.shoot_client.call_async(req)
        future.add_done_callback(self.shoot_response_callback)

    def shoot_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"服务端响应: success={response.success}, message='{response.message}'")
        except Exception as e:
            self.get_logger().error(f"调用服务端失败: {e}")

    def parse_target_data(self, data_str):
        """
        解析目标数据
        输入格式: "p,272,252" 
        返回: (x, y) 或 None
        """
        try:
            # 移除可能的空白字符
            data_str = data_str.strip()
            
            # 使用正则表达式解析数据
            # 匹配 p,x,y 或 c,x,y,r 格式
            pattern = r'^([p]),(\d+),(\d+)(?:,(\d+))?$'
            match = re.match(pattern, data_str)
            
            if match:
                data_type = match.group(1)  # 'p' 或 'c'
                x = int(match.group(2))
                y = int(match.group(3))
                
                self.get_logger().debug(f'解析目标数据成功: 类型={data_type}, 坐标=({x}, {y})')
                return (x, y)
            # else:
            #     self.get_logger().warning(f'目标数据格式不正确: {data_str}')
            #     return None
                
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
        格式: "@0,x偏差,y偏差\"
        """
        # 格式化指令字符串
        command = f"@0,{x_error},{y_error}\r"
        return command
    
    def target_callback(self, msg):
        """目标数据回调函数"""
        try:
            target_center = self.parse_target_data(msg.data)
            
            if target_center is None:
                self.error_count += 1
                self.success_counter = 0
                self.get_logger().warning(f'无法解析目标数据: {msg.data}')
                return
            
            x_error, y_error = self.calculate_error(target_center)
            if x_error is None or y_error is None:
                self.success_counter = 0
                return
            
            # 误差判定
            if abs(x_error) < self.error_threshold and abs(y_error) < self.error_threshold:
                self.success_counter += 1
                if self.success_counter >= self.success_required_count:
                    self.get_logger().info("连续命中阈值满足，发送打靶成功信号")
                    self.call_shoot_service(True)
                    self.success_counter = 0
            else:
                self.success_counter = 0
            
            # 更新最新的云台控制指令（将由定时器发送）
            self.latest_gimbal_command = self.format_gimbal_command(x_error, y_error)
            
            self.last_target_center = target_center
            self.get_logger().info(
                f'目标: {target_center} | 误差: ({x_error:+d},{y_error:+d}) | 成功计数:{self.success_counter}'
            )
            
        except Exception as e:
            self.error_count += 1
            self.success_counter = 0
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
