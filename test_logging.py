#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志功能测试脚本
模拟ROS2节点输出，测试启动脚本的日志记录功能
"""

import time
import sys
import random

def simulate_node_output(node_name, duration=10):
    """模拟节点输出"""
    print(f"[INFO] [{node_name}] 节点启动")
    
    for i in range(duration):
        # 模拟不同类型的日志输出
        log_types = ["INFO", "DEBUG", "WARN", "ERROR"]
        log_type = random.choice(log_types)
        
        messages = {
            "INFO": [
                f"处理第 {i+1} 帧图像",
                f"发布目标数据: c,320,240,95",
                f"检测到矩形目标",
                f"FPS: {random.randint(25, 35)}"
            ],
            "DEBUG": [
                f"矩形检测耗时: {random.randint(10, 50)}ms",
                f"圆形检测耗时: {random.randint(15, 45)}ms",
                f"总处理耗时: {random.randint(30, 80)}ms"
            ],
            "WARN": [
                "图像质量较低",
                "检测精度可能受影响",
                "光照条件不佳"
            ],
            "ERROR": [
                "图像处理错误",
                "无法读取摄像头数据",
                "目标检测失败"
            ]
        }
        
        message = random.choice(messages[log_type])
        print(f"[{log_type}] [{node_name}] {message}")
        
        # 随机延迟
        time.sleep(random.uniform(0.5, 2.0))
    
    print(f"[INFO] [{node_name}] 节点正常退出")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        node_name = sys.argv[1]
    else:
        node_name = "test_node"
    
    try:
        simulate_node_output(node_name, 5)  # 运行5秒
    except KeyboardInterrupt:
        print(f"[INFO] [{node_name}] 收到中断信号，正在退出...")