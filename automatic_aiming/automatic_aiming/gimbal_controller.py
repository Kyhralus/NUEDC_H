#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
import re
import time
import wiringpi
from wiringpi import GPIO
import time

class GimbalController(Node):
    """云台控制器节点"""
    
    def __init__(self):
        super().__init__('gimbal_controller')
        
        # 图像中心配置 - 可灵活修改
        self.image_width = 960
        self.image_height = 720
        self.image_center_x = self.image_width // 2   # 400
        self.image_center_y = self.image_height // 2  # 300
        
        # 阈值配置
        self.error_threshold = 5           # 最大允许误差 (用于云台控制的误差判定)
        self.success_required_count = 10   # 连续成功次数要求 (用于云台控制的成功计数)
        self.success_counter = 0           # 计数器 (用于云台控制的成功计数)

        # 激光开启判定配置
        self.laser_error_threshold = 20     # 激光开启的最大允许误差
        self.laser_success_required_count = 3 # 激光开启所需的连续成功次数
        self.laser_counter = 0             # 激光计数器
        
        # 激光控制相关配置
        self.laser_pin = 9                # GPIO9用于控制激光 (根据用户提供的wiringpi示例)
        
        # 激光安全控制开关
        self.laser_switch_pin = 5        # GPIO5用于激光安全开关(输入模式)
        self.laser_safety_enabled = False # 激光安全状态（True=允许激光开启，False=禁止激光开启）
        
        # 激光状态管理
        self.laser_mode = "auto_off"      # 激光模式: "auto_off" 或 "continuous"
        self.laser_opened = False         # 激光状态标志
        self.laser_shooting_time = 1.0    # 激光打开时间(1.0s)
        self.laser_timer = None           # 激光定时器
        
        # 初始化GPIO
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(self.laser_pin, GPIO.OUTPUT)
        wiringpi.digitalWrite(self.laser_pin, GPIO.LOW)  # 初始状态为关闭
        
        # 设置激光安全开关引脚为输入模式，默认上拉
        wiringpi.pinMode(self.laser_switch_pin, GPIO.INPUT)
        wiringpi.pullUpDnControl(self.laser_switch_pin, GPIO.PUD_UP)  # 上拉电阻
        
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

        # 创建定时器用于检查激光安全开关状态 (每0.5秒检查一次)
        self.laser_safety_timer = self.create_timer(0.7, self.check_laser_safety_switch)

        self.get_logger().info(f'云台控制器节点已启动')
        self.get_logger().info(f'图像中心设置为: ({self.image_center_x}, {self.image_center_y})')
        self.get_logger().info(f'激光模式: {self.laser_mode} (auto_off=瞄准后1秒关闭, continuous=持续开启)')
        self.get_logger().info(f'激光安全开关: GPIO{self.laser_switch_pin} (低电平:禁用激光, 高电平:允许激光)')
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
            
            # 误差判定 (用于云台控制)
            if abs(x_error) < self.error_threshold and abs(y_error) < self.error_threshold:
                self.success_counter += 1
                # if self.success_counter >= self.success_required_count:
                #     self.get_logger().info("连续命中阈值满足")
                #     self.success_counter = 0
            else:
                self.success_counter = 0
            
            # 激光开启判定
            if abs(x_error) <= self.laser_error_threshold and abs(y_error) <= self.laser_error_threshold:
                self.laser_counter += 1
                if self.laser_counter >= self.laser_success_required_count:   # 判定打中
                    self.get_logger().info(f"连续{self.laser_success_required_count}次误差小于等于{self.laser_error_threshold}，激光可开启")
                    # 根据激光模式控制激光
                    if self.laser_mode == "auto_off":
                        self.control_laser_auto_off()
                    elif self.laser_mode == "continuous":
                        self.control_laser_continuous(True)
                    self.laser_counter = 0 # 满足条件后重置计数器

                    # 发送打中的状态个小车
                    cmd_msg = String()
                    cmd_msg.data = f"@1\r"
                    self.gimbal_publisher.publish(cmd_msg)
                    self.get_logger().debug(f'发送云台指令: {cmd_msg.data}')


            else:
                self.laser_counter = 0 # 不满足条件则重置计数器
            
            # 更新最新的云台控制指令（将由定时器发送）
            self.latest_gimbal_command = self.format_gimbal_command(x_error, y_error)
            
            self.last_target_center = target_center
            self.get_logger().info(
                f'目标: {target_center} | 误差: ({x_error:+d},{y_error:+d}) | 成功计数:{self.success_counter} | 激光状态:{self.laser_opened}'
            )
            
        except Exception as e:
            self.error_count += 1
            self.success_counter = 0
            self.get_logger().error(f'处理目标数据时出错: {str(e)}')

    def check_laser_safety_switch(self):
        """每1秒检查一次激光安全开关状态"""
        switch_state = wiringpi.digitalRead(self.laser_switch_pin)
        old_state = self.laser_safety_enabled
        
        # 更新安全状态
        self.laser_safety_enabled = (switch_state == GPIO.HIGH)
        
        # 记录状态变化
        if self.laser_safety_enabled != old_state:
            if self.laser_safety_enabled:
                self.get_logger().info("激光安全开关已切换到高电平，允许激光开启")
            else:
                self.get_logger().info("激光安全开关已切换到低电平，禁止激光开启")
        
        # 如果安全开关为低电平且激光当前是开启状态，则强制关闭激光
        if not self.laser_safety_enabled and self.laser_opened:
            wiringpi.digitalWrite(self.laser_pin, GPIO.LOW)
            self.laser_opened = False
            self.get_logger().info("激光安全开关处于低电平，强制关闭激光")
            
            # 如果有定时器，取消它
            if self.laser_timer:
                self.laser_timer.cancel()
                self.laser_timer = None
    
    def is_laser_allowed(self):
        """检查当前是否允许激光开启"""
        # 使用缓存的安全状态而不是每次都读取GPIO
        return self.laser_safety_enabled
    
    def control_laser_auto_off(self):
        """自动关闭模式：开启激光1秒后自动关闭"""
        # 先检查安全开关是否允许激光开启
        if not self.is_laser_allowed():
            self.get_logger().info("激光安全开关处于低电平，禁止开启激光")
            return
            
        if not self.laser_opened:
            # 开启激光 (高电平是开)
            wiringpi.digitalWrite(self.laser_pin, GPIO.HIGH)
            self.laser_opened = True
            self.get_logger().info(f"激光已开启 (自动关闭模式，{self.laser_shooting_time}秒后关闭)")
            
            # 取消之前的定时器（如果存在）
            if self.laser_timer:
                self.laser_timer.cancel()
            
            # 创建定时器，延时关闭激光
            self.laser_timer = self.create_timer(
                self.laser_shooting_time, 
                self.close_laser_auto
            )
            self.get_logger().info(f"定时器已创建，将在 {self.laser_shooting_time} 秒后调用 close_laser_auto")
    
    def control_laser_continuous(self, turn_on: bool):
        """持续开启模式：手动控制激光开关"""
        if turn_on:
            # 先检查安全开关是否允许激光开启
            if not self.is_laser_allowed():
                self.get_logger().info("激光安全开关处于低电平，禁止开启激光")
                return
                
            if not self.laser_opened:
                # 开启激光
                wiringpi.digitalWrite(self.laser_pin, GPIO.HIGH)
                self.laser_opened = True
                self.get_logger().info("激光已开启 (持续开启模式)")
        elif self.laser_opened:
            # 关闭激光
            wiringpi.digitalWrite(self.laser_pin, GPIO.LOW)
            self.laser_opened = False
            self.get_logger().info("激光已关闭 (持续开启模式)")
    
    def close_laser_auto(self):
        """自动关闭激光（用于自动关闭模式）"""
        self.get_logger().info("尝试关闭激光...")
        if self.laser_opened:
            wiringpi.digitalWrite(self.laser_pin, GPIO.LOW)
            self.laser_opened = False
            self.get_logger().info("激光已自动关闭")
            
            # 清理定时器，确保只执行一次
            if self.laser_timer:
                self.laser_timer.cancel()
                self.laser_timer = None
            self.get_logger().info("定时器已取消并清空。")
        else:
            self.get_logger().info("激光已处于关闭状态或未开启，无需操作。")
    
    
    def laser_mode_service_callback(self, request, response):
        """激光模式切换服务回调函数"""
        try:
            if request.data:
                # 收到True时，切换到continuous模式
                old_mode = self.laser_mode
                self.laser_mode = "continuous"
                
                # 如果从自动模式切换到持续模式，需要处理当前状态
                if old_mode == "auto_off" and self.laser_timer:
                    self.laser_timer.cancel()
                    self.laser_timer = None
                
                self.get_logger().info(f"激光模式已通过服务切换: {old_mode} -> continuous")
                response.success = True
                response.message = f"激光模式已切换为continuous"
            else:
                # 收到False时，切换到auto_off模式
                old_mode = self.laser_mode
                self.laser_mode = "auto_off"
                
                # 如果从持续模式切换到自动模式，需要关闭激光
                if old_mode == "continuous" and self.laser_opened:
                    self.control_laser_continuous(False)
                
                self.get_logger().info(f"激光模式已通过服务切换: {old_mode} -> auto_off")
                response.success = True
                response.message = f"激光模式已切换为auto_off"
                
        except Exception as e:
            self.get_logger().error(f"激光模式切换服务出错: {str(e)}")
            response.success = False
            response.message = f"激光模式切换失败: {str(e)}"
        
        return response

    def switch_laser_mode(self, new_mode: str):
        """切换激光模式"""
        if new_mode in ["auto_off", "continuous"]:
            old_mode = self.laser_mode
            self.laser_mode = new_mode
            
            # 如果从持续模式切换到自动模式，需要关闭激光
            if old_mode == "continuous" and new_mode == "auto_off" and self.laser_opened:
                self.control_laser_continuous(False)
            
            self.get_logger().info(f"激光模式已切换: {old_mode} -> {new_mode}")
        else:
            self.get_logger().error(f"无效的激光模式: {new_mode}，支持的模式: auto_off, continuous")
    
    def get_laser_status(self):
        """获取激光状态信息"""
        return {
            'mode': self.laser_mode,
            'opened': self.laser_opened,
            'auto_off_time': self.laser_shooting_time,
            'safety_enabled': self.laser_safety_enabled
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
            # 取消激光定时器
            if hasattr(gimbal_controller, 'laser_timer') and gimbal_controller.laser_timer:
                gimbal_controller.laser_timer.cancel()
            if hasattr(gimbal_controller, 'laser_safety_timer'):
                gimbal_controller.laser_safety_timer.cancel()
            
            gimbal_controller.destroy_node()
            # 确保激光关闭
            wiringpi.digitalWrite(gimbal_controller.laser_pin, GPIO.LOW)
            print("激光已安全关闭")
        rclpy.shutdown()

if __name__ == '__main__':
    main()
