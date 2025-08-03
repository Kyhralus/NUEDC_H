#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成主控制器节点
合并了原来的 main_controller 和 gimbal_controller 功能
减少服务通信，提高响应速度
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from std_srvs.srv import SetBool

# 导入工具模块
from .utils.task_manager import TaskManager
from .utils.gimbal_control import GimbalControl
from .utils.communication_manager import CommunicationManager


class IntegratedMainController(Node):
    """集成主控制器类"""
    
    def __init__(self):
        super().__init__('main_controller')
        
        # 初始化各个管理器
        self.task_manager = TaskManager(self.get_logger())
        self.gimbal_control = GimbalControl(self.get_logger())
        self.comm_manager = CommunicationManager(self)
        
        # 设置回调函数
        self.comm_manager.set_uart_command_callback(self.handle_uart_command)
        self.comm_manager.set_target_data_callback(self.handle_target_data)
        
        # 创建定时器
        self.laser_control_timer = self.create_timer(1.0, self.control_laser_output)
        
        # 创建激光模式服务（保持兼容性）
        self.laser_mode_service = self.create_service(
            SetBool, 'switch_laser_mode', self.laser_mode_service_callback
        )
        
        self.get_logger().info("集成主控制器节点初始化完成")
        self.get_logger().info("已合并原 main_controller 和 gimbal_controller 功能")
    
    def handle_uart_command(self, msg):
        """处理串口指令 - 高实时性模式"""
        try:
            # 使用任务管理器处理指令
            task_name, actions = self.task_manager.process_command(msg.data)
            
            # 立即发布任务状态到 /uart3_sender_topic（非阻塞）
            if task_name:
                self.comm_manager.publish_task_status(task_name)
            
            # 快速处理相关动作
            for action_type, enable in actions:
                # 检查是否需要跳过重复调用
                if self.task_manager.should_skip_service_call(action_type, enable):
                    continue
                
                # 异步调用服务（非阻塞）
                if self.comm_manager.call_service_async(action_type, enable):
                    self.task_manager.update_service_state(action_type, enable)
                
        except Exception as e:
            self.get_logger().error(f"处理串口指令异常: {str(e)}")
            self.comm_manager.publish_task_status("ERROR")
    
    def handle_target_data(self, msg):
        """处理目标数据 - 高实时性模式"""
        try:
            # 使用云台控制器处理目标数据
            commands = self.gimbal_control.process_target_data(msg.data)
            
            # 立即发布云台控制指令到 /uart1_sender_topic（非阻塞）
            if commands:
                self.comm_manager.publish_gimbal_commands(commands)
                
        except Exception as e:
            self.get_logger().error(f"处理目标数据异常: {str(e)}")
    
    def control_laser_output(self):
        """定时器回调：控制激光输出"""
        try:
            self.gimbal_control.control_laser_output()
        except Exception as e:
            self.get_logger().error(f"控制激光输出时发生异常: {str(e)}")
    
    def laser_mode_service_callback(self, request, response):
        """激光模式切换服务回调函数"""
        try:
            # 保持服务接口兼容性
            self.get_logger().info("激光模式服务调用 - 功能已简化为瞄准即开启")
            response.success = True
            response.message = "激光模式已简化为瞄准即开启"
                
        except Exception as e:
            self.get_logger().error(f"激光模式切换服务出错: {str(e)}")
            response.success = False
            response.message = f"激光模式切换失败: {str(e)}"
        
        return response
    
    def get_system_status(self):
        """获取系统状态"""
        return {
            'task_manager': self.task_manager.get_status(),
            'gimbal_control': self.gimbal_control.get_status(),
            'communication': self.comm_manager.get_status()
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            self.gimbal_control.cleanup()
            if hasattr(self, 'laser_control_timer'):
                self.laser_control_timer.cancel()
            self.get_logger().info("资源清理完成")
        except Exception as e:
            self.get_logger().error(f"清理资源时出错: {str(e)}")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    
    try:
        node = IntegratedMainController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("用户终止程序")
    except ExternalShutdownException:
        if node:
            node.get_logger().info("外部关闭信号，正常退出")
    except Exception as e:
        if node:
            node.get_logger().error(f"节点运行出错: {str(e)}")
        else:
            print(f"节点初始化失败: {str(e)}")
    finally:
        if node:
            try:
                node.cleanup()
                node.destroy_node()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()