#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
云台控制模块
负责目标跟踪、误差计算和激光控制
"""

import re
import time
from periphery import GPIO


class GimbalControl:
    """云台控制类"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # 图像中心配置
        self.image_width = 960
        self.image_height = 720
        self.image_center_x = self.image_width // 2   # 480
        self.image_center_y = self.image_height // 2  # 360
        
        # 阈值配置
        self.error_threshold = 5           # 最大允许误差
        self.success_required_count = 10   # 连续成功次数要求
        self.success_counter = 0           # 计数器

        # 激光开启判定配置
        self.laser_error_threshold = 20     # 激光开启的最大允许误差
        self.laser_success_required_count = 3 # 激光开启所需的连续成功次数
        self.laser_counter = 0             # 激光计数器
        
        # 激光模式和状态管理
        self.laser_mode = "continuous"      # 激光模式
        self.shotted = 0                   # 瞄准状态: 0=未瞄准, 1=已瞄准靶子
        
        # 初始化GPIO
        try:
            self.key_in = GPIO("/dev/gpiochip1", 22, "in")       # 外部开关
            self.laser_out = GPIO("/dev/gpiochip1", 27, "out")   # 激光输出
            self.laser_out.write(False)  # 初始化激光为关闭状态
            self.gpio_initialized = True
            self.logger.info("GPIO初始化成功")
        except Exception as e:
            self.logger.error(f"GPIO初始化失败: {str(e)}")
            self.gpio_initialized = False
        
        # 误差统计
        self.error_count = 0
        self.last_target_center = None
        self.latest_gimbal_command = None
        
        self.logger.info(f'云台控制模块初始化完成')
        self.logger.info(f'图像中心: ({self.image_center_x}, {self.image_center_y})')
    
    def parse_target_data(self, data_str):
        """
        解析目标数据
        输入格式: "p,272,252" 
        返回: (x, y) 或 None
        """
        try:
            data_str = data_str.strip()
            pattern = r'^([p]),(\d+),(\d+)(?:,(\d+))?$'
            match = re.match(pattern, data_str)
            
            if match:
                data_type = match.group(1)
                x = int(match.group(2))
                y = int(match.group(3))
                
                self.logger.debug(f'解析目标数据成功: 类型={data_type}, 坐标=({x}, {y})')
                return (x, y)
                
        except Exception as e:
            self.logger.error(f'解析目标数据时出错: {data_str}, 错误: {str(e)}')
            return None
    
    def calculate_error(self, target_center):
        """
        计算目标中心与图像中心的误差
        返回: (x_error, y_error)
        """
        if target_center is None:
            return None, None
        
        target_x, target_y = target_center
        x_error = target_x - self.image_center_x
        y_error = target_y - self.image_center_y
        
        return -x_error, y_error
    
    def format_gimbal_command(self, x_error, y_error):
        """格式化云台控制指令"""
        return f"@0,{x_error},{y_error}\r"
    
    def process_target_data(self, data_str):
        """
        处理目标数据，返回需要发送的指令列表
        返回: [command1, command2, ...]
        """
        commands = []
        
        try:
            target_center = self.parse_target_data(data_str)
            
            if target_center is None:
                self.error_count += 1
                self.success_counter = 0
                self.logger.warning(f'无法解析目标数据: {data_str}')
                return commands
            
            x_error, y_error = self.calculate_error(target_center)
            if x_error is None or y_error is None:
                self.success_counter = 0
                return commands
            
            # 误差判定
            if abs(x_error) < self.error_threshold and abs(y_error) < self.error_threshold:
                self.success_counter += 1
            else:
                self.success_counter = 0
            
            # 激光开启判定
            if abs(x_error) <= self.laser_error_threshold and abs(y_error) <= self.laser_error_threshold:
                self.laser_counter += 1
                if self.laser_counter >= self.laser_success_required_count:
                    self.logger.info(f"连续{self.laser_success_required_count}次误差小于等于{self.laser_error_threshold}，已瞄准靶子")
                    self.shotted = 1
                    self.laser_counter = 0
                    
                    # 添加打中状态指令
                    commands.append("@1\r")
            else:
                self.laser_counter = 0
                self.shotted = 0
            
            # 添加云台控制指令
            gimbal_cmd = self.format_gimbal_command(x_error, y_error)
            commands.append(gimbal_cmd)
            
            self.last_target_center = target_center
            self.logger.info(
                f'目标: {target_center} | 误差: ({x_error:+d},{y_error:+d}) | 成功计数:{self.success_counter} | 瞄准状态:{self.shotted}'
            )
            
        except Exception as e:
            self.error_count += 1
            self.success_counter = 0
            self.shotted = 0
            self.logger.error(f'处理目标数据时出错: {str(e)}')
        
        return commands
    
    def control_laser_output(self):
        """控制激光输出"""
        if not self.gpio_initialized:
            return
        
        try:
            # 读取外部开关状态
            key_state = self.key_in.read()
            
            if key_state == 0:  # 低电平，关闭激光
                self.laser_out.write(False)
                self.logger.debug("外部开关为低电平，激光已关闭")
            else:  # 高电平，根据瞄准状态控制激光
                if self.shotted == 1:
                    self.laser_out.write(True)
                    self.logger.debug("已瞄准靶子，激光开启")
                else:
                    self.laser_out.write(False)
                    self.logger.debug("未瞄准靶子，激光关闭")
                    
        except Exception as e:
            self.logger.error(f"控制激光输出时出错: {str(e)}")
            try:
                self.laser_out.write(False)
            except:
                pass
    
    def cleanup(self):
        """清理GPIO资源"""
        if self.gpio_initialized:
            try:
                self.laser_out.write(False)
                self.laser_out.close()
                self.key_in.close()
                self.logger.info("GPIO资源已清理")
            except Exception as e:
                self.logger.error(f"清理GPIO资源时出错: {str(e)}")
    
    def get_status(self):
        """获取云台控制状态"""
        return {
            'image_center': (self.image_center_x, self.image_center_y),
            'image_size': (self.image_width, self.image_height),
            'last_target': self.last_target_center,
            'error_count': self.error_count,
            'shotted': self.shotted,
            'success_counter': self.success_counter,
            'laser_counter': self.laser_counter
        }