#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCP图像发送模块
提供简单的API用于通过TCP协议发送OpenCV图像
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import time


class ImageSender:
    """
    图像发送器类
    用于将OpenCV图像通过TCP发送到指定IP地址
    """
    
    def __init__(self, ip='127.0.0.1', port=8485):
        """
        初始化发送器
        
        参数:
            ip (str): 接收端IP地址
            port (int): 接收端端口号
        """
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """
        建立TCP连接
        
        返回:
            bool: 连接是否成功
        """
        try:
            if self.socket:
                self.disconnect()
                
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            print(f"已连接到 {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开TCP连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            print("连接已断开")
    
    def send_image(self, image, quality=90, retry=1):
        """
        发送OpenCV图像
        
        参数:
            image (numpy.ndarray): OpenCV图像
            quality (int): JPEG压缩质量 (1-100)
            retry (int): 连接失败时的重试次数
            
        返回:
            bool: 发送是否成功
        """
        if image is None:
            print("错误: 图像为空")
            return False
            
        # 如果未连接，尝试连接
        if not self.connected:
            if not self.connect():
                if retry <= 0:
                    return False
                print(f"尝试重新连接... (剩余尝试次数: {retry})")
                time.sleep(1)
                return self.send_image(image, quality, retry-1)
        
        try:
            # 将图像编码为JPEG格式以减小数据大小
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', image, encode_param)
            
            # 序列化图像数据
            data = pickle.dumps(encoded_img)
            
            # 发送数据大小信息，使用struct打包为网络字节序
            msg_size = struct.pack("!L", len(data))
            self.socket.sendall(msg_size)
            
            # 发送图像数据
            self.socket.sendall(data)
            return True
            
        except Exception as e:
            print(f"发送失败: {e}")
            self.connected = False
            self.socket = None
            
            # 如果有重试次数，尝试重新连接并发送
            if retry > 0:
                print(f"尝试重新发送... (剩余尝试次数: {retry})")
                time.sleep(1)
                return self.send_image(image, quality, retry-1)
            return False
    
    def __del__(self):
        """析构函数，确保连接被关闭"""
        self.disconnect()


# 全局发送器实例，用于简化API调用
_default_sender = None

def init_sender(ip='127.0.0.1', port=8485):
    """
    初始化默认图像发送器
    
    参数:
        ip (str): 接收端IP地址
        port (int): 接收端端口号
        
    返回:
        ImageSender: 发送器实例
    """
    global _default_sender
    _default_sender = ImageSender(ip, port)
    return _default_sender

def send_image(image, quality=90):
    """
    使用默认发送器发送图像
    
    参数:
        image (numpy.ndarray): OpenCV图像
        quality (int): JPEG压缩质量 (1-100)
        
    返回:
        bool: 发送是否成功
    """
    global _default_sender
    if _default_sender is None:
        _default_sender = ImageSender()
    
    return _default_sender.send_image(image, quality)

def disconnect():
    """断开默认发送器的连接"""
    global _default_sender
    if _default_sender:
        _default_sender.disconnect()


# 使用示例
# if __name__ == "__main__":
#     # 创建一个测试图像
#     test_img = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.putText(test_img, "TCP图像测试", (50, 240), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # 方法1: 使用全局函数
#     # init_sender('127.0.0.1', 8485)
#     # send_image(test_img)
#     # disconnect()
    
#     # 方法2: 使用ImageSender类
#     sender = ImageSender('192.168.31.89', 5000)
#     sender.connect()
#     sender.send_image(test_img)
#     sender.disconnect()