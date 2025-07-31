#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import time
import threading
import os
from collections import deque
from datetime import datetime

class EKFFilter(Node):
    """
    扩展卡尔曼滤波节点，用于对目标点进行预测
    
    订阅：/target_data 话题，获取目标点坐标
    发布：/ekf_data 话题，发布预测的目标点坐标
    """
    
    def __init__(self):
        super().__init__('ekf_filter_node')
        
        # 创建订阅者和发布者
        self.target_subscriber = self.create_subscription(
            String,
            '/target_data',
            self.target_callback,
            10
        )
        
        self.prediction_publisher = self.create_publisher(
            String,
            '/ekf_data',
            10
        )
        
        # 预测时间（秒）
        self.prediction_time = 0.3
        
        # 帧率设置 - 默认30帧/秒
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps  # 约0.033秒
        
        # 预测下一帧的时间
        self.next_frame_prediction_time = self.frame_interval
        
        # 初始化状态变量
        self.initialized = False
        self.last_timestamp = None
        
        # 状态向量 [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # 状态协方差矩阵
        self.P = np.eye(4) * 100
        
        # 过程噪声协方差
        self.Q = np.eye(4)
        self.Q[0, 0] = 0.01  # x位置噪声
        self.Q[1, 1] = 0.01  # y位置噪声
        self.Q[2, 2] = 0.1   # x速度噪声
        self.Q[3, 3] = 0.1   # y速度噪声
        
        # 测量噪声协方差
        self.R = np.eye(2) * 5.0
        
        # 存储历史数据 - 使用deque限制长度
        max_history = 500
        self.history = {
            'time': deque(maxlen=max_history),
            'measured_x': deque(maxlen=max_history),
            'measured_y': deque(maxlen=max_history),
            'filtered_x': deque(maxlen=max_history),
            'filtered_y': deque(maxlen=max_history),
            'predicted_x': deque(maxlen=max_history),
            'predicted_y': deque(maxlen=max_history)
        }
        
        # 数据锁
        self.data_lock = threading.Lock()
        
        # 创建保存目录
        self.save_dir = os.path.join(os.getcwd(), 'ekf_plots')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建定时器，每3秒保存一次图表
        self.save_timer = self.create_timer(3.0, self.save_trajectory_plot)
        
        self.get_logger().info('EKF Filter Node Started with Plot Saving Every 3 Seconds')
        self.get_logger().info(f'Plot save directory: {self.save_dir}')
    
    def target_callback(self, msg):
        """处理接收到的目标点数据"""
        data = msg.data.split(',')
        
        # 检查数据格式是否正确
        if len(data) != 3 or data[0] != 'p':
            return
        
        try:
            # 解析目标点坐标
            x = float(data[1])
            y = float(data[2])
            
            # 获取当前时间
            current_time = time.time()
            
            # 如果是第一次接收数据，初始化状态
            if not self.initialized:
                self.state[0] = x
                self.state[1] = y
                self.state[2] = 0.0  # 初始速度为0
                self.state[3] = 0.0
                self.initialized = True
                self.last_timestamp = current_time
                
                # 记录测量值和滤波值
                self.record_history(current_time, x, y, x, y, x, y)
                return
            
            # 计算时间差
            dt = current_time - self.last_timestamp
            self.last_timestamp = current_time
            
            if dt <= 0:
                return
            
            # 预测步骤
            predicted_state, predicted_P = self.predict(dt)
            
            # 更新步骤
            measurement = np.array([x, y])
            self.update(measurement)
            
            # 预测未来位置
            future_state, _ = self.predict(self.prediction_time, update_state=False)
            future_x = future_state[0]
            future_y = future_state[1]
            
            # 预测下一帧位置 (1/30秒后，约0.033秒)
            next_frame_state, _ = self.predict(self.next_frame_prediction_time, update_state=False)
            next_frame_x = next_frame_state[0]
            next_frame_y = next_frame_state[1]
            
            # 打印坐标信息
            self.print_coordinate_info(x, y, self.state[0], self.state[1], 
                                     next_frame_x, next_frame_y, future_x, future_y)
            
            # 发布预测结果
            self.publish_prediction(future_x, future_y)
            
            # 记录历史数据
            self.record_history(
                current_time, 
                x, y,
                self.state[0], self.state[1],
                future_x, future_y
            )
            
        except Exception as e:
            self.get_logger().error(f'处理目标数据时出错: {str(e)}')
    
    def predict(self, dt, update_state=True):
        """
        预测步骤
        
        参数:
            dt: 时间差
            update_state: 是否更新状态
            
        返回:
            predicted_state: 预测后的状态
            predicted_P: 预测后的协方差
        """
        # 状态转移矩阵
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 预测状态
        predicted_state = F @ self.state
        
        # 预测协方差
        predicted_P = F @ self.P @ F.T + self.Q
        
        if update_state:
            self.state = predicted_state
            self.P = predicted_P
        
        return predicted_state, predicted_P
    
    def update(self, measurement):
        """
        更新步骤
        
        参数:
            measurement: 测量值 [x, y]
        """
        # 测量矩阵
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 计算卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 计算测量残差
        y = measurement - H @ self.state
        
        # 更新状态
        self.state = self.state + K @ y
        
        # 更新协方差
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P
    
    def publish_prediction(self, x, y):
        """发布预测结果"""
        msg = String()
        msg.data = f"pre,{x:.2f},{y:.2f}"
        self.prediction_publisher.publish(msg)
    
    def save_trajectory_plot(self):
        """每3秒保存一次轨迹图"""
        with self.data_lock:
            if len(self.history['time']) < 2:
                return
            
            try:
                # 转换为列表
                times = list(self.history['time'])
                measured_x = list(self.history['measured_x'])
                measured_y = list(self.history['measured_y'])
                filtered_x = list(self.history['filtered_x'])
                filtered_y = list(self.history['filtered_y'])
                predicted_x = list(self.history['predicted_x'])
                predicted_y = list(self.history['predicted_y'])
                
                if not measured_x or not measured_y:
                    return
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.save_dir, f'ekf_trajectory_{timestamp}.png')
                
                # 创建图表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                fig.suptitle(f'EKF Tracking Results - {timestamp}', fontsize=16)
                
                # 轨迹图
                ax1.set_title('Target Trajectory')
                ax1.set_xlabel('X Coordinate (pixels)')
                ax1.set_ylabel('Y Coordinate (pixels)')
                ax1.grid(True, alpha=0.3)
                
                # 绘制轨迹
                ax1.plot(measured_x, measured_y, 'r.', markersize=4, label='Measured Points', alpha=0.7)
                ax1.plot(filtered_x, filtered_y, 'b-', linewidth=2, label='Filtered Trajectory')
                
                # 绘制最近5个预测点
                if len(predicted_x) >= 5 and len(predicted_y) >= 5:
                    ax1.scatter(predicted_x[-5:], predicted_y[-5:], c='g', s=60, marker='x', 
                               label='Recent Predictions', linewidth=2)
                
                # 标记起始和结束点
                if measured_x and measured_y:
                    ax1.scatter(measured_x[0], measured_y[0], c='orange', s=100, marker='o', 
                               label='Start Point')
                    ax1.scatter(measured_x[-1], measured_y[-1], c='red', s=100, marker='s', 
                               label='Current Point')
                
                ax1.legend()
                ax1.set_xlim(0, 800)
                ax1.set_ylim(0, 600)
                ax1.invert_yaxis()  # 图像坐标系Y轴向下
                
                # 时间序列图
                relative_time = [t - times[0] for t in times]
                
                ax2.set_title('Coordinates vs Time')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Coordinate Value (pixels)')
                ax2.grid(True, alpha=0.3)
                
                # 绘制X坐标时间序列
                ax2.plot(relative_time, measured_x, 'r.', markersize=3, label='Measured X', alpha=0.6)
                ax2.plot(relative_time, filtered_x, 'b-', linewidth=1.5, label='Filtered X')
                
                # 绘制Y坐标时间序列
                ax2.plot(relative_time, measured_y, 'r^', markersize=3, label='Measured Y', alpha=0.6)
                ax2.plot(relative_time, filtered_y, 'g-', linewidth=1.5, label='Filtered Y')
                
                ax2.legend()
                
                # 添加统计信息文字
                stats_text = f'Points: {len(measured_x)}\n'
                stats_text += f'Time Span: {relative_time[-1]:.1f}s\n'
                if len(filtered_x) > 1:
                    speed_x = np.std(np.diff(filtered_x))
                    speed_y = np.std(np.diff(filtered_y))
                    stats_text += f'X Variation: {speed_x:.2f}\n'
                    stats_text += f'Y Variation: {speed_y:.2f}'
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                self.get_logger().info(f'Trajectory plot saved: {filename}')
                
                # 清理旧图片 - 只保留最近20张
                self.cleanup_old_plots()
                
            except Exception as e:
                self.get_logger().error(f'Failed to save trajectory plot: {str(e)}')
    
    def cleanup_old_plots(self):
        """清理旧的图片文件，只保留最近20张"""
        try:
            plot_files = []
            for filename in os.listdir(self.save_dir):
                if filename.startswith('ekf_trajectory_') and filename.endswith('.png'):
                    filepath = os.path.join(self.save_dir, filename)
                    plot_files.append((filepath, os.path.getctime(filepath)))
            
            # 按创建时间排序
            plot_files.sort(key=lambda x: x[1])
            
            # 删除多余的文件，只保留最近20张
            if len(plot_files) > 20:
                for filepath, _ in plot_files[:-20]:
                    try:
                        os.remove(filepath)
                        self.get_logger().info(f'Cleaned up old plot: {os.path.basename(filepath)}')
                    except Exception as e:
                        self.get_logger().warning(f'Failed to remove old plot {filepath}: {e}')
                        
        except Exception as e:
            self.get_logger().warning(f'Plot cleanup failed: {str(e)}')
    
    def record_history(self, timestamp, measured_x, measured_y, filtered_x, filtered_y, predicted_x, predicted_y):
        """记录历史数据"""
        with self.data_lock:
            self.history['time'].append(timestamp)
            self.history['measured_x'].append(measured_x)
            self.history['measured_y'].append(measured_y)
            self.history['filtered_x'].append(filtered_x)
            self.history['filtered_y'].append(filtered_y)
            self.history['predicted_x'].append(predicted_x)
            self.history['predicted_y'].append(predicted_y)

    def print_coordinate_info(self, measured_x, measured_y, filtered_x, filtered_y, 
                            next_frame_x, next_frame_y, future_x, future_y):
        """打印坐标信息"""
        self.get_logger().info("=" * 80)
        self.get_logger().info("COORDINATE TRACKING INFORMATION")
        self.get_logger().info("=" * 80)
        
        # 当前测量点
        self.get_logger().info(f"📍 Current Measured Point:    ({measured_x:.2f}, {measured_y:.2f})")
        
        # 过滤后的点
        self.get_logger().info(f"🔍 Filtered Point:           ({filtered_x:.2f}, {filtered_y:.2f})")
        
        # 预测下一帧的点 (1/30秒后)
        self.get_logger().info(f"⏭️  Next Frame Prediction:    ({next_frame_x:.2f}, {next_frame_y:.2f}) [+{self.next_frame_prediction_time:.3f}s]")
        
        # 预测未来的点 (0.3秒后)
        self.get_logger().info(f"🎯 Future Prediction:        ({future_x:.2f}, {future_y:.2f}) [+{self.prediction_time:.1f}s]")
        
        # 计算偏差
        filter_error_x = filtered_x - measured_x
        filter_error_y = filtered_y - measured_y
        self.get_logger().info(f"📊 Filter Error:             ({filter_error_x:+.2f}, {filter_error_y:+.2f})")
        
        # 计算预测位移
        next_frame_displacement_x = next_frame_x - filtered_x
        next_frame_displacement_y = next_frame_y - filtered_y
        self.get_logger().info(f"🚀 Next Frame Displacement:  ({next_frame_displacement_x:+.2f}, {next_frame_displacement_y:+.2f})")
        
        # 计算当前速度 (像素/秒)
        velocity_x = self.state[2]
        velocity_y = self.state[3]
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        self.get_logger().info(f"⚡ Current Velocity:         ({velocity_x:.2f}, {velocity_y:.2f}) px/s, Speed: {speed:.2f} px/s")
        
        self.get_logger().info("=" * 80)

def main(args=None):
    rclpy.init(args=args)
    
    ekf_node = EKFFilter()
    
    try:
        rclpy.spin(ekf_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 保存最终轨迹图
        ekf_node.save_trajectory_plot()
        ekf_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()