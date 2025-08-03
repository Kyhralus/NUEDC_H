#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通信管理器模块
负责ROS2通信接口的管理
"""

from std_msgs.msg import String
from std_srvs.srv import SetBool
import time


class CommunicationManager:
    """通信管理器类"""
    
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        
        # 初始化发布器
        self.task_status_pub = node.create_publisher(
            String, "/uart3_sender_topic", 10
        )
        
        self.gimbal_publisher = node.create_publisher(
            String, "/uart1_sender_topic", 10
        )
        
        # 初始化订阅器
        self.uart_subscriber = node.create_subscription(
            String, "/uart1_receiver_topic",
            self._uart_callback, 10
        )
        
        self.target_subscriber = node.create_subscription(
            String, "/target_data",
            self._target_callback, 10
        )
        
        # 初始化服务客户端（保留兼容性，但可能不使用）
        self.laser_mode_client = node.create_client(
            SetBool, 'switch_laser_mode'
        )
        
        self.perspective_client = node.create_client(
            SetBool, 'set_perspective_publish'
        )
        
        self.target_detection_client = node.create_client(
            SetBool, 'start_target_detection'
        )
        
        # 回调函数引用
        self.uart_command_callback = None
        self.target_data_callback = None
        
        # 非阻塞初始化，立即完成
        self.logger.info("通信管理器初始化完成 - 非阻塞模式")
        
        # 异步检查发布器连接状态（不阻塞）
        self._check_publishers_async()
    
    def set_uart_command_callback(self, callback):
        """设置串口指令回调函数"""
        self.uart_command_callback = callback
    
    def set_target_data_callback(self, callback):
        """设置目标数据回调函数"""
        self.target_data_callback = callback
    
    def _uart_callback(self, msg):
        """内部串口回调函数"""
        if self.uart_command_callback:
            self.uart_command_callback(msg)
    
    def _target_callback(self, msg):
        """内部目标数据回调函数"""
        if self.target_data_callback:
            self.target_data_callback(msg)
    
    def publish_task_status(self, task_name):
        """发布任务状态到 /uart3_sender_topic"""
        try:
            # 确保发布器已准备好
            if self.task_status_pub.get_subscription_count() == 0:
                self.logger.warn(f"没有订阅者连接到 /uart3_sender_topic，但仍然发布: {task_name}")
            
            msg = String(data=task_name)
            self.task_status_pub.publish(msg)
            self.logger.info(f"已发布任务状态到 /uart3_sender_topic: {task_name}")
            
            # 确保消息被发送
            time.sleep(0.02)
            return True
            
        except Exception as e:
            self.logger.error(f"发布任务状态时出错: {str(e)}")
            return False
    
    def publish_gimbal_command(self, command):
        """发布云台控制指令到 /uart1_sender_topic - 非阻塞"""
        try:
            msg = String(data=command)
            self.gimbal_publisher.publish(msg)
            self.logger.info(f'已发布云台指令到 /uart1_sender_topic: {command}')
            return True
            
        except Exception as e:
            self.logger.error(f"发布云台指令时出错: {str(e)}")
            return False
    
    def publish_gimbal_commands(self, commands):
        """批量发布云台控制指令 - 非阻塞高速模式"""
        if not commands:
            return 0
            
        success_count = 0
        # 高速批量发布，不添加延迟
        for cmd in commands:
            if self.publish_gimbal_command(cmd):
                success_count += 1
        
        self.logger.debug(f"云台指令批量发布: {success_count}/{len(commands)} 成功")
        return success_count
    
    def _check_publishers_async(self):
        """异步检查发布器状态 - 非阻塞"""
        try:
            task_subs = self.task_status_pub.get_subscription_count()
            gimbal_subs = self.gimbal_publisher.get_subscription_count()
            
            self.logger.debug(f"发布器状态 - 任务状态: {task_subs} 订阅者, 云台指令: {gimbal_subs} 订阅者")
        except Exception as e:
            self.logger.debug(f"检查发布器状态时出错: {str(e)}")
    
    def call_service_async(self, service_type, enable):
        """异步调用服务（保留兼容性）"""
        try:
            client = None
            service_name = ""
            
            if service_type == 'target_detection':
                client = self.target_detection_client
                service_name = "目标检测服务"
            elif service_type == 'perspective':
                client = self.perspective_client
                service_name = "透视变换服务"
            elif service_type == 'laser_mode':
                client = self.laser_mode_client
                service_name = "激光模式服务"
            else:
                self.logger.error(f"未知服务类型: {service_type}")
                return False
            
            # 快速检查服务可用性
            if not client.wait_for_service(timeout_sec=0.5):
                self.logger.warn(f'{service_name}暂时不可用，跳过调用')
                return False
            
            request = SetBool.Request()
            request.data = enable
            
            # 异步调用，不等待结果
            future = client.call_async(request)
            action = "启动" if enable else "停止"
            self.logger.info(f'已发送{service_name}{action}请求')
            return True
            
        except Exception as e:
            self.logger.error(f'调用{service_name}时出错: {str(e)}')
            return False
    
    def get_status(self):
        """获取通信管理器状态"""
        return {
            'task_status_subscribers': self.task_status_pub.get_subscription_count(),
            'gimbal_subscribers': self.gimbal_publisher.get_subscription_count(),
            'uart_callback_set': self.uart_command_callback is not None,
            'target_callback_set': self.target_data_callback is not None
        }