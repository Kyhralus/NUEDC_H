#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
import re
import time
from periphery import GPIO

class GimbalController(Node):
    """云台控制器节点"""
    
    def __init__(self):
        super().__init__('gimbal_controller')
        
        # 图像中心配置 - 可灵活修改
        self.image_width = 960
        self.image_height = 720
        self.image_center_x = self.image_width // 2   # 480
        self.image_center_y = self.image_height // 2  # 360
        
        # 阈值配置
        self.error_threshold = 5           # 最大允许误差 (用于云台控制的误差判定)
        self.success_required_count = 10   # 连续成功次数要求 (用于云台控制的成功计数)
        self.success_counter = 0           # 计数器 (用于云台控制的成功计数)

        # 激光开启判定配置
        self.laser_error_threshold = 20     # 激光开启的最大允许误差
        self.laser_success_required_count = 3 # 激光开启所需的连续成功次数
        self.laser_counter = 0             # 激光计数器
        
        # 激光模式和状态管理
        self.laser_mode = "continuous"      # 激光模式: "auto_off" 或 "continuous"
        self.shotted = 0                   # 瞄准状态: 0=未瞄准, 1=已瞄准靶子
        
        # 初始化GPIO对象
        self.key_in = GPIO("/dev/gpiochip1", 22, "in")       # gpio 54 ---> 54 - 32 = 22 (外部开关)
        self.laser_out = GPIO("/dev/gpiochip1", 27, "out")   # gpio 59 ---> 59 - 32 = 27 (激光输出)
        
        # 初始化激光输出为False (布尔值)
        self.laser_out.write(False)
        
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
        
        # 创建服务端 - 激光模式切换服务
        self.laser_mode_service = self.create_service(
            SetBool,
            'switch_laser_mode',
            self.laser_mode_service_callback
        )

        # 误差统计
        self.error_count = 0
        self.last_target_center = None
        
        # 存储最新的云台控制指令
        self.latest_gimbal_command = None
        
        # 创建30Hz定时器用于发送云台控制指令
        self.gimbal_timer = self.create_timer(0.02, self.gimbal_timer_callback)

        # 创建定时器用于检查外部开关状态 (每1秒检查一次)
        self.laser_control_timer = self.create_timer(1.0, self.control_laser_output)

        self.get_logger().info(f'云台控制器节点已启动')
        self.get_logger().info(f'图像中心设置为: ({self.image_center_x}, {self.image_center_y})')
        self.get_logger().info(f'激光模式: {self.laser_mode} (auto_off=瞄准后开启, continuous=持续开启)')
        self.get_logger().info(f'外部开关: GPIO22 (低电平:关闭激光, 高电平:允许激光)')
        self.get_logger().info(f'订阅话题: /target_data')
        self.get_logger().info(f'发布话题: /uart1_sender_topic (30Hz)')
        self.get_logger().info(f'控制频率: 30Hz')
    
    def gimbal_timer_callback(self):
        """30Hz定时器回调函数，用于发送云台控制指令"""
        if self.latest_gimbal_command is not None:
            cmd_msg = String()
            cmd_msg.data = self.latest_gimbal_command
            self.gimbal_publisher.publish(cmd_msg)
            self.latest_gimbal_command = None   # 及时更新为 None
            self.get_logger().debug(f'发送云台指令: {cmd_msg.data}')

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
        
        return -x_error, y_error
    
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
            
            # 误差判定 (用于云台控制)
            if abs(x_error) < self.error_threshold and abs(y_error) < self.error_threshold:
                self.success_counter += 1
            else:
                self.success_counter = 0
            
            # 激光开启判定
            if abs(x_error) <= self.laser_error_threshold and abs(y_error) <= self.laser_error_threshold:
                self.laser_counter += 1
                if self.laser_counter >= self.laser_success_required_count:   # 判定打中
                    self.get_logger().info(f"连续{self.laser_success_required_count}次误差小于等于{self.laser_error_threshold}，已瞄准靶子")
                    self.shotted = 1  # 设置瞄准状态
                    self.laser_counter = 0 # 满足条件后重置计数器

                    # 发送打中的状态给小车
                    cmd_msg = String()
                    cmd_msg.data = f"@1\r"
                    self.gimbal_publisher.publish(cmd_msg)
                    self.get_logger().debug(f'发送云台指令: {cmd_msg.data}')
            else:
                self.laser_counter = 0 # 不满足条件则重置计数器
                self.shotted = 0  # 重置瞄准状态
            
            # 更新最新的云台控制指令（将由定时器发送）
            self.latest_gimbal_command = self.format_gimbal_command(x_error, y_error)
            
            self.last_target_center = target_center
            self.get_logger().info(
                f'目标: {target_center} | 误差: ({x_error:+d},{y_error:+d}) | 成功计数:{self.success_counter} | 瞄准状态:{self.shotted}'
            )
            
        except Exception as e:
            self.error_count += 1
            self.success_counter = 0
            self.shotted = 0  # 重置瞄准状态
            self.get_logger().error(f'处理目标数据时出错: {str(e)}')

    def control_laser_output(self):
        """每1秒检查外部开关状态并控制激光输出"""
        try:
            # 读取外部开关状态
            key_state = self.key_in.read()
            
            if key_state == 0:  # 低电平，没有电压通过
                self.laser_out.write(False)  # 激光笔直接关闭 (使用布尔值)
                self.get_logger().debug("外部开关为低电平，激光已关闭")
            else:  # 高电平，有电压通过
                # 简化模式：只要瞄准了靶子就开启激光，否则关闭
                if self.shotted == 1:
                    self.laser_out.write(True)  # 使用布尔值True
                    self.get_logger().debug("已瞄准靶子，激光开启")
                else:
                    self.laser_out.write(False)  # 使用布尔值False
                    self.get_logger().debug("未瞄准靶子，激光关闭")
                    
        except Exception as e:
            self.get_logger().error(f"控制激光输出时出错: {str(e)}")
            # 出错时安全关闭激光
            try:
                self.laser_out.write(False)  # 使用布尔值False
            except:
                pass

    def laser_mode_service_callback(self, request, response):
        """激光模式切换服务回调函数 - 保持兼容性但功能简化"""
        try:
            # 保持服务接口兼容性，但实际功能已简化为只要瞄准就开启
            if request.data:
                self.get_logger().info("激光模式服务调用 - 功能已简化为瞄准即开启")
                response.success = True
                response.message = "激光模式已简化为瞄准即开启"
            else:
                self.get_logger().info("激光模式服务调用 - 功能已简化为瞄准即开启")
                response.success = True
                response.message = "激光模式已简化为瞄准即开启"
                
        except Exception as e:
            self.get_logger().error(f"激光模式切换服务出错: {str(e)}")
            response.success = False
            response.message = f"激光模式切换失败: {str(e)}"
        
        return response

    def switch_laser_mode(self, new_mode: str):
        """切换激光模式 - 保持接口兼容性"""
        self.get_logger().info("激光模式已简化为瞄准即开启，模式切换功能已禁用")
    
    def get_laser_status(self):
        """获取激光状态信息"""
        return {
            'mode': 'aim_on',  # 新模式：瞄准即开启
            'shotted': self.shotted,
            'laser_output': self.shotted  # 激光输出状态等于瞄准状态
        }

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
    gimbal_controller = None
    
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
        # 安全清理资源
        if gimbal_controller is not None:
            try:
                # 关闭激光输出
                if hasattr(gimbal_controller, 'laser_out'):
                    gimbal_controller.laser_out.write(False)  # 使用布尔值False
                    gimbal_controller.laser_out.close()
                    print("激光输出已关闭")
                
                # 关闭外部开关GPIO
                if hasattr(gimbal_controller, 'key_in'):
                    gimbal_controller.key_in.close()
                    print("外部开关GPIO已关闭")
                
                # 取消定时器
                if hasattr(gimbal_controller, 'laser_control_timer'):
                    gimbal_controller.laser_control_timer.cancel()
                
                if hasattr(gimbal_controller, 'gimbal_timer'):
                    gimbal_controller.gimbal_timer.cancel()
                
                # 销毁节点
                gimbal_controller.destroy_node()
                
                print("所有资源已清理")
                
            except Exception as cleanup_error:
                print(f"清理资源时出错: {str(cleanup_error)}")
        
        rclpy.shutdown()

if __name__ == '__main__':
    main()