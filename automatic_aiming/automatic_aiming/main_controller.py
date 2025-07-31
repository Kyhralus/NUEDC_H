import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2
import numpy as np

# 任务状态枚举（与题目要求对应）
class TaskState:
    # 基本要求
    BASIC_AUTO_DRIVE = 10  # 基本要求1：自动寻迹行驶 --- IGNORE
    BASIC_AIM_FREE = 20    # 基本要求2：自由位置瞄准靶心
    BASIC_AIM_FIXED = 30   # 基本要求3：指定位置瞄准靶心
    
    # 发挥部分
    ADVANCED_TARGET_TRACK = 40  # 发挥部分1：靶心跟踪 --- 一圈， 发挥部分2：靶心跟踪 --- 两圈
    ADVANCED_DRAW_CIRCLE = 50  # 发挥部分3：同步画圆
    ADVANCED_DIY = 60  # 发挥部分4：自定义

class AimingController(Node):
    def __init__(self):
        super().__init__('aiming_controller')
        
        # 回调组设置
        self.reentrant_group = ReentrantCallbackGroup()
        self.exclusive_group = MutuallyExclusiveCallbackGroup()
        
        # 系统状态
        self.current_task = None
        self.task_running = False
        
        # 数据存储
        self.target_center = None  # 靶心坐标
        self.target_circle = None  #  6cm 靶环
        
        # 初始化功能模块
        self.aim_module = AimModule(self)        # 瞄准模块（二维云台+激光笔）
        self.vision_module = VisionModule(self)  # 视觉识别模块
        
        # 初始化通信
        self.init_communications()
        
        self.get_logger().info("瞄准系统主控节点初始化完成")

    def init_communications(self):
        """初始化ROS2通信接口"""
        # 订阅串口指令（接收控制命令）
        self.create_subscription(
            String, "uart_commands",
            self.handle_uart_command, 10,
            callback_group=self.reentrant_group
        )
        
        # 发布系统状态
        self.status_pub = self.create_publisher(
            String, "system_status", 10
        )
        
        # 发布瞄准结果
        self.aim_result_pub = self.create_publisher(
            String, "aim_results", 10
        )
        
        # 服务客户端（控制瞄准模块）
        self.aim_client = self.create_client(
            SetBool, "enable_aim",
            callback_group=self.exclusive_group
        )

    def handle_uart_command(self, msg):
        """处理串口指令（映射到对应任务）"""
        cmd = msg.data.strip()
        self.get_logger().info(f"收到指令: {cmd}")
        
        # 指令-任务映射（专注于瞄准相关任务）
        command_map = {
            "r10": TaskState.BASIC_AUTO_DRIVE,  # 基本要求1：自动寻迹行驶 --- IGNORE
            "r20": TaskState.BASIC_AIM_FREE,
            "r30": TaskState.BASIC_AIM_FIXED,
            "r40": TaskState.ADVANCED_TARGET_TRACK,
            "r50": TaskState.ADVANCED_DRAW_CIRCLE,
            "r60": TaskState.ADVANCED_DIY
        }
        
        if cmd in command_map:
            self.switch_task(command_map[cmd])
        else:
            self.get_logger().warn(f"未知指令: {cmd}")
            self.aim_result_pub.publish(String(data=f"@ERROR,未知指令:{cmd}"))

    def switch_task(self, new_task):
        """切换任务状态并执行"""
        if self.task_running:
            self.get_logger().info(f"停止当前任务: {self.current_task}")
            self.aim_module.stop_aiming()
        
        self.current_task = new_task
        self.task_running = True
        threading.Thread(
            target=self.execute_task,
            args=(new_task,),
            daemon=True
        ).start()

    def execute_task(self, task):
        """执行指定任务"""
        try:
            if task == TaskState.BASIC_AUTO_DRIVE:
                # 基本要求2：自由位置2s内瞄准靶心（D₁≤2cm）
                self.get_logger().info("执行基本要求2：自由位置瞄准")
                # 跳过

            if task == TaskState.BASIC_AIM_FREE:
                # 基本要求2：自由位置2s内瞄准靶心（D₁≤2cm）
                self.get_logger().info("执行基本要求2：自由位置瞄准")

                
            elif task == TaskState.BASIC_AIM_FIXED:
                # 基本要求3：指定位置4s内瞄准靶心（D₁≤2cm）
                self.get_logger().info("执行基本要求3：指定位置瞄准")
  
            elif task == TaskState.ADVANCED_TARGET_TRACK:
                # 发挥部分1，2：目标跟踪（持续瞄准移动靶心），一圈+两圈
                self.get_logger().info("执行发挥部分1：目标跟踪")


            elif task == TaskState.ADVANCED_DRAW_CIRCLE:
                # 发挥部分3：同步画圆（6cm半径，同步误差<1/4圈）
                self.get_logger().info("执行发挥部分3：同步画圆")

        except Exception as e:
            self.get_logger().info(f"任务执行错误: {str(e)}")
        finally:
            self.task_running = False

    def handle_image_input(self, msg):
        """处理摄像头图像（用于靶心识别）"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 根据当前任务类型选择不同的识别模式
            if self.current_task == TaskState.ADVANCED_TARGET_TRACK:
                self.targets = self.vision_module.detect_multi_targets(self.latest_image)
            else:
                self.target_center = self.vision_module.detect_single_target(self.latest_image)
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")

class AimModule:
    """瞄准模块（二维云台+激光笔）"""
    def __init__(self, node):
        self.node = node
        self.gimbal_ready = True  # 二维云台就绪状态
        self.laser_on = False     # 激光笔状态
        self.current_pan = 0.0    # 水平角度
        self.current_tilt = 0.0   # 垂直角度
        self.target_lost_count = 0  # 目标丢失计数

    def aim_target(self, time_limit, max_distance, fixed_position=False):
        """
        瞄准单个靶心
        :param time_limit: 时间限制（秒）
        :param max_distance: 允许最大偏差（cm）
        :param fixed_position: 是否在指定位置启动
        :return: (是否成功, 最终偏差)
        """
        start_time = time.time()
        self.laser_on = True
        self.node.get_logger().info(f"激光笔开启，开始瞄准（限时{time_limit}s）")
        
        # 如果是固定位置瞄准，先复位云台
        if fixed_position:
            self.reset_gimbal()
            time.sleep(0.5)
        
        while True:
            # 检查时间是否超时
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                self.laser_on = False
                return (False, 999.9)  # 超时返回失败
            
            # 检查目标是否存在
            if self.node.target_center is None:
                self.target_lost_count += 1
                if self.target_lost_count > 20:  # 连续20帧丢失目标
                    self.laser_on = False
                    return (False, 999.9)
                time.sleep(0.05)
                continue
            
            self.target_lost_count = 0  # 重置丢失计数
            
            # 计算偏差 (实际应根据相机参数转换为厘米)
            img_center = (self.node.latest_image.shape[1]//2, 
                         self.node.latest_image.shape[0]//2)
            error_x = self.node.target_center[0] - img_center[0]
            error_y = self.node.target_center[1] - img_center[1]
            distance = np.sqrt(error_x**2 + error_y**2) * 0.1  # 转换为厘米
            
            # 控制云台移动
            self.current_pan -= error_x * 0.01  # 比例控制
            self.current_tilt += error_y * 0.01  # 比例控制
            self.current_pan = np.clip(self.current_pan, -45, 45)  # 限制角度范围
            self.current_tilt = np.clip(self.current_tilt, -30, 30)
            self.control_gimbal(self.current_pan, self.current_tilt)
            
            # 检查是否达到精度要求
            if distance <= max_distance:
                self.node.get_logger().info(f"瞄准完成，偏差: {distance:.1f}cm，耗时: {elapsed:.1f}s")
                self.laser_on = False
                return (True, distance)
            
            time.sleep(0.05)  # 控制循环频率

    def track_target(self, duration, required_accuracy):
        """
        跟踪移动目标
        :param duration: 跟踪持续时间(秒)
        :param required_accuracy: 要求的准确率(%)
        :return: (是否成功, 实际准确率)
        """
        start_time = time.time()
        self.laser_on = True
        success_count = 0
        total_count = 0
        
        self.node.get_logger().info(f"开始跟踪目标，持续{duration}s")
        
        while time.time() - start_time < duration:
            if self.node.target_center is None:
                time.sleep(0.05)
                continue
                
            # 计算当前偏差
            img_center = (self.node.latest_image.shape[1]//2, 
                         self.node.latest_image.shape[0]//2)
            error_x = self.node.target_center[0] - img_center[0]
            error_y = self.node.target_center[1] - img_center[1]
            distance = np.sqrt(error_x**2 + error_y**2) * 0.1  # 转换为厘米
            
            # 移动云台跟踪目标
            self.current_pan -= error_x * 0.02  # 稍快的响应速度
            self.current_tilt += error_y * 0.02
            self.current_pan = np.clip(self.current_pan, -45, 45)
            self.current_tilt = np.clip(self.current_tilt, -30, 30)
            self.control_gimbal(self.current_pan, self.current_tilt)
            
            # 记录准确率
            total_count += 1
            if distance <= 3.0:  # 跟踪允许更大偏差
                success_count += 1
                
            time.sleep(0.05)
        
        self.laser_on = False
        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        self.node.get_logger().info(f"跟踪结束，准确率: {accuracy:.1f}%")
        return (accuracy >= required_accuracy, accuracy)

    def aim_multi_targets(self, target_count, max_per_target):
        """
        瞄准多个目标
        :param target_count: 目标数量
        :param max_per_target: 单个目标最大瞄准时间
        :return: (是否成功, 平均瞄准时间)
        """
        self.laser_on = True
        success = True
        total_time = 0.0
        completed = 0
        
        self.node.get_logger().info(f"开始多目标瞄准，共{target_count}个目标")
        
        while completed < target_count:
            # 等待检测到足够的目标
            if len(self.node.targets) < target_count:
                self.node.get_logger().info(f"等待检测到{target_count}个目标...")
                time.sleep(1.0)
                continue
                
            # 瞄准第n个目标
            target_idx = completed
            start_time = time.time()
            target_acquired = False
            
            while time.time() - start_time < max_per_target:
                # 检查目标是否仍然存在
                if len(self.node.targets) <= target_idx:
                    break
                    
                # 计算偏差
                img_center = (self.node.latest_image.shape[1]//2, 
                             self.node.latest_image.shape[0]//2)
                error_x = self.node.targets[target_idx][0] - img_center[0]
                error_y = self.node.targets[target_idx][1] - img_center[1]
                distance = np.sqrt(error_x**2 + error_y**2) * 0.1
                
                # 移动云台
                self.current_pan -= error_x * 0.01
                self.current_tilt += error_y * 0.01
                self.current_pan = np.clip(self.current_pan, -45, 45)
                self.current_tilt = np.clip(self.current_tilt, -30, 30)
                self.control_gimbal(self.current_pan, self.current_tilt)
                
                # 检查是否瞄准成功
                if distance <= 2.0:
                    target_time = time.time() - start_time
                    total_time += target_time
                    completed += 1
                    target_acquired = True
                    self.node.get_logger().info(f"目标{completed}/{target_count}瞄准成功，耗时{target_time:.1f}s")
                    break
                    
                time.sleep(0.05)
                
            if not target_acquired:
                self.node.get_logger().error(f"目标{completed+1}瞄准超时")
                success = False
                break
        
        self.laser_on = False
        avg_time = total_time / completed if completed > 0 else 0
        return (success, avg_time)

    def draw_circle(self, radius, time_limit, max_error):
        """
        控制激光画圆
        :param radius: 圆半径(cm)
        :param time_limit: 时间限制
        :param max_error: 最大允许误差(圈)
        :return: (是否成功, 实际误差)
        """
        start_time = time.time()
        self.laser_on = True
        self.reset_gimbal()
        time.sleep(1.0)  # 等待云台复位
        
        # 圆参数
        angular_speed = 0.1  # 角速度(rad/s)
        current_angle = 0.0
        center_pan = 0.0     # 圆心水平角度
        center_tilt = 0.0    # 圆心垂直角度
        
        # 计算角度范围 (根据半径和云台特性)
        angle_range = radius * 0.5  # 角度范围与半径成正比
        
        self.node.get_logger().info(f"开始画圆，半径{radius}cm，限时{time_limit}s")
        
        while time.time() - start_time < time_limit:
            # 计算当前角度对应的云台位置
            pan = center_pan + np.cos(current_angle) * angle_range
            tilt = center_tilt + np.sin(current_angle) * angle_range
            
            # 控制云台移动
            self.control_gimbal(pan, tilt)
            
            # 更新角度
            current_angle += angular_speed
            elapsed_time = time.time() - start_time
            expected_angle = angular_speed * elapsed_time
            angle_error = abs(current_angle - expected_angle) / (2 * np.pi)  # 转换为圈数误差
            
            # 检查误差是否过大
            if angle_error > max_error:
                self.node.get_logger().error(f"画圆误差过大: {angle_error:.2f}圈")
                self.laser_on = False
                return (False, angle_error)
            
            # 检查是否完成一圈
            if current_angle >= 2 * np.pi:
                self.node.get_logger().info(f"画圆完成，耗时{elapsed_time:.1f}s，误差{angle_error:.2f}圈")
                self.laser_on = False
                return (True, angle_error)
            
            time.sleep(0.05)
        
        # 超时未完成
        final_error = abs(current_angle - (angular_speed * time_limit)) / (2 * np.pi)
        self.laser_on = False
        return (False, final_error)

    def control_gimbal(self, pan_angle, tilt_angle):
        """控制云台移动到指定角度"""
        # 实际应用中应替换为硬件控制代码
        # 发送角度指令到云台控制器
        self.node.get_logger().debug(f"云台控制: 水平={pan_angle:.1f}°, 垂直={tilt_angle:.1f}°")

    def reset_gimbal(self):
        """重置云台到初始位置"""
        self.node.get_logger().info("云台复位中...")
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.control_gimbal(0.0, 0.0)

    def stop_aiming(self):
        """停止瞄准，关闭激光"""
        self.laser_on = False
        self.node.get_logger().info("停止瞄准，激光关闭")

class VisionModule:
    """视觉识别模块（处理靶面图像）"""
    def __init__(self, node):
        self.node = node
        # 初始化目标检测参数（红色靶心）
        self.hsv_lower = np.array([0, 120, 70])    # 红色下限
        self.hsv_upper = np.array([10, 255, 255])  # 红色上限
        self.min_radius = 5
        self.max_radius = 50

    def detect_single_target(self, image):
        """识别单个靶心位置"""
        if image is None:
            return None
            
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 提取红色区域
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 5)  # 去噪
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 找到最大的圆形轮廓（假设是靶心）
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        
        # 过滤不合理的大小
        if radius < self.min_radius or radius > self.max_radius:
            return None
            
        # 计算轮廓的圆形度（判断是否为圆形）
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 只保留圆形度高的目标
        if circularity > 0.7:
            return (int(x), int(y))
        return None

    def detect_multi_targets(self, image):
        """识别多个靶心位置"""
        if image is None:
            return []
            
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 提取红色区域
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 5)  # 去噪
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        targets = []
        for contour in contours:
            # 计算最小外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # 过滤大小
            if radius < self.min_radius or radius > self.max_radius:
                continue
                
            # 计算圆形度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity > 0.7:
                targets.append((int(x), int(y)))
        
        # 按x坐标排序目标
        targets.sort(key=lambda p: p[0])
        return targets

def main(args=None):
    rclpy.init(args=args)
    node = AimingController()
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
