from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo
import os
from datetime import datetime

def generate_launch_description():
    # 获取当前时间戳用于日志文件名
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 日志目录
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 定义所有节点
    nodes = [
        # 摄像头发布节点
        Node(
            package='automatic_aiming',
            executable='camera_publisher',
            name='camera_publisher',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # 主控制器节点
        Node(
            package='automatic_aiming',
            executable='main_controller',
            name='main_controller',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # 目标检测节点
        Node(
            package='automatic_aiming',
            executable='target_detect',
            name='target_detect',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # UART1通信节点
        Node(
            package='automatic_aiming',
            executable='uart1_receiver',
            name='uart1_receiver',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        Node(
            package='automatic_aiming',
            executable='uart1_sender',
            name='uart1_sender',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # UART3语音模块节点
        Node(
            package='automatic_aiming',
            executable='uart3_receiver',
            name='uart3_receiver',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        Node(
            package='automatic_aiming',
            executable='uart3_sender',
            name='uart3_sender',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # 语音节点
        Node(
            package='automatic_aiming',
            executable='speakr_node',
            name='speakr_node',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        # UART0备用节点
        Node(
            package='automatic_aiming',
            executable='uart0_receiver',
            name='uart0_receiver',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
        
        Node(
            package='automatic_aiming',
            executable='uart0_sender',
            name='uart0_sender',
            output='both',  # 同时输出到终端和日志
            arguments=['--ros-args', '--log-level', 'INFO']
        ),
    ]
    
    # 添加启动信息
    launch_actions = [
        LogInfo(msg=f'启动自动瞄准系统 - 时间戳: {timestamp}'),
        LogInfo(msg=f'日志保存目录: {log_dir}'),
    ]
    
    # 添加所有节点
    launch_actions.extend(nodes)
    
    return LaunchDescription(launch_actions)
