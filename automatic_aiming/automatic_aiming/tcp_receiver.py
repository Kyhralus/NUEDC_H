#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCP图像接收模块
用于接收通过TCP协议发送的OpenCV图像
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import threading


class ImageReceiver:
    """
    图像接收器类
    用于接收通过TCP发送的OpenCV图像
    """
    
    def __init__(self, port=8485, display=True):
        """
        初始化接收器
        
        参数:
            port (int): 监听端口号
            display (bool): 是否显示接收到的图像
        """
        self.port = port
        self.display = display
        self.socket = None
        self.client_socket = None
        self.running = False
        self.thread = None
        self.latest_image = None
        self.image_callback = None
    
    def start(self, callback=None):
        """
        启动接收服务
        
        参数:
            callback (callable): 接收到图像时的回调函数，接收图像作为参数
            
        返回:
            bool: 启动是否成功
        """
        if self.running:
            print("接收器已在运行")
            return False
            
        self.image_callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """停止接收服务"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        
        if self.thread:
            self.thread.join(timeout=1.0)
            
        self.socket = None
        self.client_socket = None
        self.thread = None
        print("接收服务已停止")
    
    def _receive_loop(self):
        """接收循环，在单独的线程中运行"""
        try:
            # 创建TCP套接字
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(5)
            print(f"接收服务已启动，监听端口: {self.port}")
            
            while self.running:
                print("等待连接...")
                self.client_socket, addr = self.socket.accept()
                print(f"已连接: {addr}")
                
                try:
                    while self.running:
                        # 接收数据大小信息 (4字节)
                        msg_size_data = self._recv_all(4)
                        if not msg_size_data:
                            break
                            
                        # 解析数据大小
                        msg_size = struct.unpack("!L", msg_size_data)[0]
                        
                        # 接收图像数据
                        data = self._recv_all(msg_size)
                        if not data:
                            break
                            
                        # 反序列化并解码图像
                        encoded_img = pickle.loads(data)
                        image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                        
                        # 保存最新图像
                        self.latest_image = image
                        
                        # 如果有回调函数，调用它
                        if self.image_callback:
                            self.image_callback(image)
                        
                        # 如果需要显示图像
                        if self.display:
                            cv2.imshow('接收的图像', image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                                
                except Exception as e:
                    print(f"接收数据时出错: {e}")
                
                finally:
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
                    
        except Exception as e:
            print(f"接收服务出错: {e}")
            
        finally:
            self.stop()
            if self.display:
                cv2.destroyAllWindows()
    
    def _recv_all(self, n):
        """
        接收指定字节数的数据
        
        参数:
            n (int): 要接收的字节数
            
        返回:
            bytes: 接收到的数据，如果连接断开则返回None
        """
        if not self.client_socket:
            return None
            
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def get_latest_image(self):
        """
        获取最新接收到的图像
        
        返回:
            numpy.ndarray: 最新的图像，如果没有则返回None
        """
        return self.latest_image
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.stop()


# 使用示例
if __name__ == "__main__":
    # 创建接收器并启动
    receiver = ImageReceiver(port=5000, display=True)
    
    # 可选：定义回调函数
    def on_image_received(img):
        if img is not None:
            # 在这里处理接收到的图像
            height, width = img.shape[:2]
            print(f"接收到图像，尺寸: {width}x{height}")
    
    # 启动接收器，并传入回调函数
    receiver.start(callback=on_image_received)
    
    try:
        # 保持程序运行
        print("按Ctrl+C停止接收服务")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # 停止接收器
        receiver.stop()