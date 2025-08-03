#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务管理器模块
负责处理不同的任务指令和状态管理
"""

class TaskState:
    """任务状态枚举"""
    # 基本要求
    BASIC_AUTO_DRIVE = 10  # 基本要求1：自动寻迹行驶
    BASIC_AIM_FREE = 20    # 基本要求2：自由位置瞄准靶心
    BASIC_AIM_FIXED = 30   # 基本要求3：指定位置瞄准靶心
    
    # 发挥部分
    ADVANCED_TARGET_TRACK_1 = 40  # 发挥部分1：靶心跟踪（一圈）
    ADVANCED_TARGET_TRACK_2 = 50  # 发挥部分2：靶心跟踪（两圈）
    ADVANCED_DRAW_CIRCLE = 60     # 发挥部分3：同步画圆
    ADVANCED_DIY = 70             # 发挥部分4：自定义


class TaskManager:
    """任务管理器类"""
    
    def __init__(self, logger):
        self.logger = logger
        self.current_task = None
        self.task_running = False
        
        # 任务状态跟踪
        self.target_detection_enabled = False
        self.perspective_enabled = False
        self.laser_mode_enabled = False
        
        # 数据存储
        self.target_center = None  # 靶心坐标
        self.target_circle = None  # 6cm靶环
    
    def process_command(self, cmd):
        """
        处理串口指令
        返回: (task_name, actions) 
        actions是需要执行的动作列表
        """
        cmd = cmd.strip()
        self.logger.info(f"处理指令: {cmd}")
        
        actions = []
        task_name = None
        
        if cmd == "r10":
            task_name = "task1"
            
        elif cmd == "r20":
            task_name = "task2"
            actions.append(('target_detection', True))

        elif cmd == "r30":
            task_name = "task3"
            actions.append(('target_detection', True))

        elif cmd == "r40":
            task_name = "task4"
            actions.append(('target_detection', True))
        
        elif cmd == "r41":
            self.logger.info("扩展任务一，激光开启")
            actions.append(('laser_mode', True))

        elif cmd == "r50":
            task_name = "task5"
            actions.append(('target_detection', True))
        
        elif cmd == "r51":
            self.logger.info("扩展任务二，激光开启")
            actions.append(('laser_mode', True))

        elif cmd == "r60":
            task_name = "task6"
            actions.append(('target_detection', True))
            actions.append(('perspective', True))
            actions.append(('laser_mode', True))
        
        elif cmd == "r61":
            self.logger.info("扩展任务三，激光开启")
            actions.append(('laser_mode', True))
          
        elif cmd == "r70":
            task_name = "task7"
            actions.append(('laser_mode', True))

        elif cmd == "r00":
            self.logger.info("收到r00指令，关闭目标检测数据发布")
            actions.append(('target_detection', False))
            task_name = "stopped"

        else:
            self.logger.warn(f"未知指令: {cmd}")
            return None, []
        
        return task_name, actions
    
    def update_service_state(self, service_type, enabled):
        """更新服务状态"""
        if service_type == 'target_detection':
            self.target_detection_enabled = enabled
        elif service_type == 'perspective':
            self.perspective_enabled = enabled
        elif service_type == 'laser_mode':
            self.laser_mode_enabled = enabled
    
    def should_skip_service_call(self, service_type, enable):
        """检查是否应该跳过服务调用（避免重复调用）"""
        if service_type == 'target_detection':
            return self.target_detection_enabled == enable
        elif service_type == 'perspective':
            return self.perspective_enabled == enable
        elif service_type == 'laser_mode':
            return self.laser_mode_enabled == enable
        return False
    
    def get_status(self):
        """获取任务管理器状态"""
        return {
            'current_task': self.current_task,
            'task_running': self.task_running,
            'target_detection_enabled': self.target_detection_enabled,
            'perspective_enabled': self.perspective_enabled,
            'laser_mode_enabled': self.laser_mode_enabled,
            'target_center': self.target_center,
            'target_circle': self.target_circle
        }