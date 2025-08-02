import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import SetBool
import threading
import time
import numpy as np # Keep numpy as it might be used elsewhere, or remove if not needed. For now, keep.

# 任务状态枚举（与题目要求对应）
class TaskState:
    # 基本要求
    BASIC_AUTO_DRIVE = 10  # 基本要求1：自动寻迹行驶 --- IGNORE
    BASIC_AIM_FREE = 20    # 基本要求2：自由位置瞄准靶心
    BASIC_AIM_FIXED = 30   # 基本要求3：指定位置瞄准靶心
    
    # 发挥部分
    ADVANCED_TARGET_TRACK = 40  # 发挥部分1：靶心跟踪 --- 一圈， 
    ADVANCED_TARGET_TRACK = 50  # 发挥部分2：靶心跟踪 --- 两圈
    ADVANCED_DRAW_CIRCLE = 60  # 发挥部分3：同步画圆
    ADVANCED_DIY = 70  # 发挥部分4：自定义

class MainController(Node): # Renamed
    def __init__(self):
        super().__init__('main_controller') # Renamed
        
        # 回调组设置
        self.reentrant_group = ReentrantCallbackGroup()
        self.exclusive_group = MutuallyExclusiveCallbackGroup()
        
        # 系统状态
        self.current_task = None
        self.task_running = False
        
        # 数据存储 - 保持，但可能不再使用
        self.target_center = None  # 靶心坐标
        self.target_circle = None  #  6cm 靶环
        
        # 初始化通信
        self.init_communications()
        
        self.get_logger().info("主控节点初始化完成") # Updated log message
    
    def call_laser_mode_service(self, enable_continuous):
        """调用激光模式切换服务"""
        if not self.laser_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('激光模式切换服务不可用')
            return False
        
        request = SetBool.Request()
        request.data = enable_continuous
        
        # 使用异步方式调用服务，避免阻塞主线程
        future = self.laser_mode_client.call_async(request)
        
        # 添加回调函数处理服务响应
        future.add_done_callback(
            lambda future: self.laser_mode_callback(future, enable_continuous)
        )
        
        return True
    
    def laser_mode_callback(self, future, enable_continuous):
        """激光模式切换服务回调函数"""
        try:
            response = future.result()
            if response.success:
                mode = "continuous" if enable_continuous else "auto_off"
                self.get_logger().info(f'激光模式切换成功: {mode} - {response.message}')
            else:
                self.get_logger().error(f'激光模式切换失败: {response.message}')
        except Exception as e:
            self.get_logger().error(f'处理激光模式切换服务响应时出错: {str(e)}')
    
    def call_target_detection_service(self, enable):
        """调用目标检测服务"""
        if not self.target_detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('目标检测服务不可用')
            return False
        
        request = SetBool.Request()
        request.data = enable
        
        # 使用异步方式调用服务，避免阻塞主线程
        future = self.target_detection_client.call_async(request)
        
        # 添加回调函数处理服务响应
        future.add_done_callback(
            lambda future: self.target_detection_callback(future, enable)
        )
        
        return True
    
    def target_detection_callback(self, future, enable):
        """目标检测服务回调函数"""
        try:
            response = future.result()
            if response.success:
                action = "启动" if enable else "停止"
                self.get_logger().info(f'目标检测服务{action}成功: {response.message}')
            else:
                self.get_logger().error(f'目标检测服务调用失败: {response.message}')
        except Exception as e:
            self.get_logger().error(f'处理目标检测服务响应时出错: {str(e)}')
    
    def call_perspective_service(self, enable):
        """调用透视变换数据发布服务"""
        if not self.perspective_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('透视变换数据发布服务不可用')
            return False
        
        request = SetBool.Request()
        request.data = enable
        
        # 使用异步方式调用服务，避免阻塞主线程
        future = self.perspective_client.call_async(request)
        
        # 添加回调函数处理服务响应
        future.add_done_callback(
            lambda future: self.perspective_callback(future, enable)
        )
        
        return True
    
    def perspective_callback(self, future, enable):
        """透视变换数据发布服务回调函数"""
        try:
            response = future.result()
            if response.success:
                action = "启动" if enable else "停止"
                self.get_logger().info(f'透视变换数据发布{action}成功: {response.message}')
            else:
                self.get_logger().error(f'透视变换数据发布服务调用失败: {response.message}')
        except Exception as e:
            self.get_logger().error(f'处理透视变换数据发布服务响应时出错: {str(e)}')
    
    def init_communications(self):
        """初始化ROS2通信接口"""
        # 订阅串口指令（接收控制命令）
        self.create_subscription(
            String, "/uart1_receiver_topic", # Changed topic
            self.handle_uart_command, 10,
            callback_group=self.reentrant_group
        )
        
        # 发布任务状态到 /uart3_sender_topic
        self.task_status_pub = self.create_publisher(
            String, "/uart3_sender_topic", 10
        )
        
        # 创建激光模式切换服务客户端
        self.laser_mode_client = self.create_client(
            SetBool,
            'switch_laser_mode'
        )
        
        # 透视变换数据发布服务客户端
        self.perspective_client = self.create_client(
            SetBool,
            'set_perspective_publish'
        )
        
        # 目标检测服务客户端
        self.target_detection_client = self.create_client(
            SetBool,
            'start_target_detection'
        )

    def handle_uart_command(self, msg):
        """处理串口指令（映射到对应任务）"""
        cmd = msg.data.strip()
        self.get_logger().info(f"收到指令: {cmd}")
        
        # 指令-任务映射，并发布到 /uart3_sender_topic
        if cmd == "r10":
            self.task_status_pub.publish(String(data="task1"))
            self.get_logger().info("发布任务状态: task1")
        elif cmd == "r20":
            self.task_status_pub.publish(String(data="task2"))
            self.get_logger().info("发布任务状态: task2")
        elif cmd == "r30":
            self.task_status_pub.publish(String(data="task3"))
            self.get_logger().info("发布任务状态: task3")

        # 发挥部分1：靶心跟踪 --- 一圈， 
        elif cmd == "r40":  # 瞄准后发送 '@1/r'
            self.task_status_pub.publish(String(data="task4"))
            self.get_logger().info("发布任务状态: task4")
            # 收到r40指令时，启动目标检测服务
            self.call_target_detection_service(True)
        
        elif cmd == "r41":
            self.get_logger().info("扩展任务一，激光开启")
            # 收到r40指令时，切换激光模式为continuous
            self.call_laser_mode_service(True)

         # 发挥部分2：靶心跟踪 --- 两圈
        elif cmd == "r50": 
            self.task_status_pub.publish(String(data="task5"))
            self.get_logger().info("发布任务状态: task4")
            # 收到r40指令时，启动目标检测服务
            self.call_target_detection_service(True)
        elif cmd == "r51": 
            self.get_logger().info("扩展任务二，激光开启")
            self.call_laser_mode_service(True)


        elif cmd == "r60": # 发挥部分3：同步画圆 
            self.task_status_pub.publish(String(data="task6"))
            self.get_logger().info("发布任务状态: task6")
            # 启动透视变换数据发布服务
            self.call_perspective_service(True)
            # 启动激光
            self.call_laser_mode_service(True)
        elif cmd == "r70": # DIY
            self.task_status_pub.publish(String(data="task7"))
            self.get_logger().info("发布任务状态: task7")
            self.call_laser_mode_service(True)
        else:
            self.get_logger().warn(f"未知指令: {cmd}")
            self.task_status_pub.publish(String(data=f"ERROR:未知指令:{cmd}"))



def main(args=None):
    rclpy.init(args=args)
    node = MainController() # Changed to MainController
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("用户终止程序")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
