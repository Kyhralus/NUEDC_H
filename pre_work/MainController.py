import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String, Float32MultiArray
from std_srvs.srv import SetBool, Empty
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from custom_msgs.srv import UpdateParameters  # 自定义服务
from custom_msgs.msg import TaskStatus  # 自定义消息
import threading
import asyncio
import time
import cv2
from cv_bridge import CvBridge

class TaskState:
    """任务状态枚举"""
    IDLE = 0
    TASK1_GIMBAL_RESET = 11
    TASK2_DIGIT_RECOGNITION = 21
    TASK3_CUP_DETECTION = 31
    TASK4_LASER_TARGETING = 41
    TASK5_NAVIGATION = 51
    TASK6_EMERGENCY_STOP = 61

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        
        # 回调组设置（支持多线程）
        self.reentrant_group = ReentrantCallbackGroup()
        self.exclusive_group = MutuallyExclusiveCallbackGroup()
        
        # 系统状态
        self.current_task = TaskState.IDLE
        self.task_running = False
        self.system_params = self.init_default_params()
        
        # 初始化各个管理器
        self.uart_manager = UartManager(self)
        self.task_manager = TaskManager(self)
        
        # ROS2通信初始化
        self.init_communications()
        
        # 数据存储
        self.latest_image = None
        self.t265_pose = None
        self.bridge = CvBridge()
        
        # 定时发送任务
        self.periodic_tasks = {}
        
        self.get_logger().info("主控节点初始化完成")

    def init_default_params(self):
        """初始化默认参数"""
        return {
            'hsv_lower': [0, 50, 50],
            'hsv_upper': [180, 255, 255],
            'circle_threshold': 50,
            'gimbal_speed': 0.5,
            't265_send_rate': 10.0  # Hz
        }

    def init_communications(self):
        """初始化ROS2通信"""
        # Topic - 订阅
        self.create_subscription(String, "uart1_receiver_topic", 
                               self.on_uart_command, 10, 
                               callback_group=self.reentrant_group)
        
        self.create_subscription(Image, "/camera/image_raw", 
                               self.on_image_received, 10,
                               callback_group=self.reentrant_group)
        
        self.create_subscription(PoseStamped, '/t265_camera/pose', 
                               self.on_t265_pose, 10,
                               callback_group=self.reentrant_group)
        
        # Topic - 发布
        self.uart_pub = self.create_publisher(String, 'uart1_sender_topic', 10)
        self.status_pub = self.create_publisher(TaskStatus, 'task_status', 10)
        
        # Service - 服务端
        self.param_service = self.create_service(UpdateParameters, 'update_parameters',
                                               self.handle_param_update,
                                               callback_group=self.exclusive_group)
        
        # Service - 客户端
        self.gimbal_client = self.create_client(SetBool, 'gimbal_control')
        self.vision_client = self.create_client(SetBool, 'vision_control')
        
        # Timer - 定时任务
        self.create_timer(0.1, self.on_t265_timer, callback_group=self.reentrant_group)  # 10Hz发送T265
        self.create_timer(1.0, self.on_status_timer, callback_group=self.reentrant_group)  # 1Hz状态发布

    def on_uart_command(self, msg):
        """串口指令回调 - 事件驱动"""
        cmd = msg.data.strip('\r\n')
        self.get_logger().info(f"收到指令: {cmd}")
        
        # 任务映射
        task_map = {
            'r11': TaskState.TASK1_GIMBAL_RESET,
            'r21': TaskState.TASK2_DIGIT_RECOGNITION,
            'r31': TaskState.TASK3_CUP_DETECTION,
            'r41': TaskState.TASK4_LASER_TARGETING,
            'r51': TaskState.TASK5_NAVIGATION,
            'r61': TaskState.TASK6_EMERGENCY_STOP
        }
        
        if cmd in task_map:
            self.switch_task(task_map[cmd])
        else:
            self.get_logger().warn(f"未知指令: {cmd}")

    def switch_task(self, new_task):
        """任务切换"""
        if self.task_running:
            self.task_manager.stop_current_task()
        
        self.current_task = new_task
        self.task_running = True
        
        # 异步启动任务
        threading.Thread(target=self.execute_task, args=(new_task,), daemon=True).start()

    def execute_task(self, task):
        """执行具体任务"""
        try:
            if task == TaskState.TASK1_GIMBAL_RESET:
                self.task_manager.execute_gimbal_reset()
            elif task == TaskState.TASK2_DIGIT_RECOGNITION:
                self.task_manager.execute_digit_recognition()
            elif task == TaskState.TASK3_CUP_DETECTION:
                self.task_manager.execute_cup_detection()
            elif task == TaskState.TASK4_LASER_TARGETING:
                self.task_manager.execute_laser_targeting()
            elif task == TaskState.TASK5_NAVIGATION:
                self.task_manager.execute_navigation()
            elif task == TaskState.TASK6_EMERGENCY_STOP:
                self.task_manager.execute_emergency_stop()
        except Exception as e:
            self.get_logger().error(f"任务执行失败: {e}")
        finally:
            self.task_running = False

    def on_image_received(self, msg):
        """图像接收回调"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def on_t265_pose(self, msg):
        """T265位姿回调"""
        self.t265_pose = msg

    def on_t265_timer(self):
        """定时发送T265数据"""
        if self.t265_pose and self.current_task == TaskState.TASK5_NAVIGATION:
            x = self.t265_pose.pose.position.x
            y = self.t265_pose.pose.position.y
            z = self.t265_pose.pose.position.z
            self.uart_manager.send_periodic_data(f"@5,{x:.3f},{y:.3f},{z:.3f}")

    def on_status_timer(self):
        """定时发布状态"""
        status = TaskStatus()
        status.current_task = self.current_task
        status.task_running = self.task_running
        self.status_pub.publish(status)

    def handle_param_update(self, request, response):
        """参数更新服务回调"""
        try:
            # 更新系统参数
            param_name = request.param_name
            param_value = request.param_value
            
            if param_name in self.system_params:
                self.system_params[param_name] = param_value
                self.get_logger().info(f"参数更新: {param_name} = {param_value}")
                response.success = True
                response.message = "参数更新成功"
            else:
                response.success = False
                response.message = f"未知参数: {param_name}"
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response

class UartManager:
    """串口通信管理器"""
    def __init__(self, node):
        self.node = node
        
    def send_result(self, data):
        """发送结果数据"""
        msg = String()
        msg.data = str(data) + '\r' if not str(data).endswith('\r') else str(data)
        self.node.uart_pub.publish(msg)
        self.node.get_logger().info(f"发送结果: {data}")
    
    def send_periodic_data(self, data):
        """发送周期性数据"""
        msg = String()
        msg.data = str(data) + '\r' if not str(data).endswith('\r') else str(data)
        self.node.uart_pub.publish(msg)

class TaskManager:
    """任务管理器"""
    def __init__(self, node):
        self.node = node
        self.current_task_thread = None
        
    def stop_current_task(self):
        """停止当前任务"""
        if self.current_task_thread and self.current_task_thread.is_alive():
            # 这里可以添加任务中断逻辑
            pass
    
    def execute_gimbal_reset(self):
        """任务1：云台复位"""
        self.node.get_logger().info("执行云台复位任务")
        # 调用云台服务
        if self.node.gimbal_client.service_is_ready():
            req = SetBool.Request()
            req.data = True
            future = self.node.gimbal_client.call_async(req)
            # 可以添加结果处理
        
        # 发送完成信号
        self.node.uart_manager.send_result("@1,completed")
    
    def execute_digit_recognition(self):
        """任务2：数字识别"""
        self.node.get_logger().info("执行数字识别任务")
        if self.node.latest_image is not None:
            # 这里添加数字识别逻辑
            result = self.process_digit_recognition(self.node.latest_image)
            self.node.uart_manager.send_result(f"@2,{result}")
    
    def execute_cup_detection(self):
        """任务3：杯子检测"""
        self.node.get_logger().info("执行杯子检测任务")
        if self.node.latest_image is not None:
            # 这里添加杯子检测逻辑
            x, y, distance = self.process_cup_detection(self.node.latest_image)
            self.node.uart_manager.send_result(f"@3,{x},{y},{distance}")
    
    def execute_laser_targeting(self):
        """任务4：激光瞄准"""
        self.node.get_logger().info("执行激光瞄准任务")
        # 这里添加激光瞄准逻辑
    
    def execute_navigation(self):
        """任务5：导航"""
        self.node.get_logger().info("执行导航任务")
        # 这里添加导航逻辑
    
    def execute_emergency_stop(self):
        """任务6：紧急停止"""
        self.node.get_logger().info("执行紧急停止")
        self.node.uart_manager.send_result("@6,emergency_stop")
    
    def process_digit_recognition(self, image):
        """数字识别处理"""
        # 使用系统参数进行图像处理
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = self.node.system_params['hsv_lower']
        upper = self.node.system_params['hsv_upper']
        # ... 数字识别逻辑
        return 5  # 示例返回值
    
    def process_cup_detection(self, image):
        """杯子检测处理"""
        # ... 杯子检测逻辑
        return 100, 200, 1.5  # 示例返回值

def main(args=None):
    rclpy.init(args=args)
    
    node = MainController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()