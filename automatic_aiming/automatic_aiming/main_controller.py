import rclpy
from rclpy.node import Node
import serial
import threading
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import math
from tf_transformations import euler_from_quaternion
import time
from std_srvs.srv import Empty  # 新增：导入服务模块


class Map():
    def __init__(self):
        # ============= 全局坐标点 ===========
        self.origin = (0, 0, 0)    # 原点 (x, y, yaw)
        self.pose = (None, None, None)      # 当前坐标
        self.point_C = (None, None, None)   # C点坐标
        self.point_B = (None, None, -90.00)   # B点坐标
        self.point_D = (None, None, 90.00)   # D点坐标

        # 目标点
        self.target_ACP = (None, None, None)    # 目标点的绝对坐标 Absolute Coordinate Point (x, y, yaw)
        self.target_RCP = (None, None, None)    # 目标点的相对坐标 Relative Coordinate Point (x, y, yaw)
        self.target_polar = (None, None)    # 目标点的极坐标(distance, angle)
        self.targets_list = [(None, None, None), (None, None, None)]    # (x,y,yaw)

        # ============== 其他点 =============
        
    
class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        # 初始化串口通信
        self.uart1_sender_init()        # 初始化发送发送，创建  uart_sender_topic 发布方
        self.uart1_receiver_init()      # 初始化发送发送，创建  uart_receiver_topic 订阅方

        # 订阅各模块结果话题
        self.lidar_sub = self.create_subscription(String, 'lidar_result', self.lidar_result_callback, 10)
        self.yolov8_sub = self.create_subscription(String, 'yolov8_result', self.yolov8_result_callback, 10)
        self.circle_laser_sub = self.create_subscription(String, 'gimbal_error', self.gimbal_error_callback, 10)
        self.t265_sub = self.create_subscription(PoseStamped, 't265_camera/pose', self.t265_pose_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, 'target_position', self.target_callback, 10)
        self.tracking_sub = self.create_subscription(PoseStamped, 'tracking', self.tracking_callback, 10)
        self.current_command = None

        # 新增：创建云台重置服务客户端
        self.gimbal_reset_client = self.create_client(Empty, 'reset_gimbal')
        # 单次检查服务是否就绪
        if not self.gimbal_reset_client.service_is_ready():
            self.get_logger().warn("云台重置服务未就绪（启动时检查），后续调用可能失败")

        # 新增：创建云台重置服务客户端
        self.gimbal_set_client = self.create_client(Empty, 'set_gimbal')
        # 单次检查服务是否就绪
        if not self.gimbal_set_client.service_is_ready():
            self.get_logger().warn("云台设置服务未就绪")

        # 新增：创建定时器
        self.timer = self.create_timer(0.005, self.loop_callback)  # 每 0.005 秒检查一次
        self.timer = self.create_timer(0.1, self.check_circle_laser_command)  # 每 0.1 秒检查一次
        self.timer = self.create_timer(0.05, self.t265_send)  # 20 Hz 发送一次

        self.map = Map()   # 创建地图实例
        # ========= 一些变量 =========
        self.digit = None
        self.enter = None
        self.cup = (None, None, None, None) # (x, y, distance, angle)
        self.shotting_result = False
        self.enter_offset = None
        self.shotting_flag = False

        
    def loop_callback(self):
        self.state_machine()

    # ========== 任务一 =========
    def lidar_result_callback(self, msg):
        '''
            更新杯子的 极坐标和笛卡尔坐标
        '''
        # self.get_logger().info(f"【雷达】杯子的位置: {msg.data}")
        parts = msg.data.split(',')
        if len(parts) >= 4:
            try:
                self.cup = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))   # (x, y, distance, angle)
                # self.get_logger().info(f"【雷达】保存杯子相对位置信息: {self.cup}")
            except (IndexError, ValueError):
                self.get_logger().error("解析杯子位置数据失败，可能包含非数字内容")
        else:
            self.get_logger().error("消息分割后元素不足，无法获取杯子位置")


    # ========== 任务二 识别数字 ========
    def yolov8_result_callback(self, msg):
        # self.get_logger().info(f"识别的数字结果: {msg.data}")
        self.digit = int(msg.data)


    # ========== 任务三 ========
    def gimbal_error_callback(self, msg):
        # 此回调函数仅在接收到消息时触发，主逻辑移到定时器回调中
        try:
            # 拆分消息并转换为浮点数
            err_pitch_str, err_yaw_str = msg.data.split(',')
            err_pitch = float(err_pitch_str)
            err_yaw = float(err_yaw_str)
            
            # 数值比较（判断误差是否在0.5以内）
            if abs(err_pitch) <= 5 and abs(err_yaw) <= 5:
                self.get_logger().info(f"------ 接收到激光打靶结果: {msg.data} ------")
                self.get_logger().info("打靶成功")
                self.shotting_result = True
                # 发送消息（简化赋值）
                # send_msg = String()
                # send_msg.data = "@4\r"  # 注意String消息需赋值给data字段
                # self.serial_send_callback(send_msg)

        except (ValueError, IndexError) as e:
            # 处理格式错误（如拆分失败、转换失败）
            self.get_logger().error(f"激光打靶结果消息格式错误: {str(e)}, 消息内容: {msg.data}")

    def check_circle_laser_command(self):
        # 从订阅的话题中获取最新消息，这里假设存在一个变量存储最新消息
        # 实际使用中需要根据具体情况修改
        # if self.current_command == 'r41':  # 激光打靶前先重置云台角度
        #     self.get_logger().info(f"------ 任务四：激光打靶，先重置云台角度 ------")
        #     self.set_gimbal_angle()  # 调用云台重置函数
        #     self.current_command = None
        pass

    def t265_pose_callback(self, msg: PoseStamped):
        # 保存坐标
        # 重组数据
        send_msg = String()
        # 换算 yaw 值
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        # 调用函数解算欧拉角
        yaw = euler_from_quaternion(quat)[2]
        # 将弧度转换为角度
        yaw_deg = math.degrees(yaw)

        # =========== 处理 =========
        # 地图构建 --- 保存当前位姿
        self.map.pose = (msg.pose.position.x, msg.pose.position.y, yaw_deg)

    def t265_send(self):
        '''
            取整扩大 100 倍 30Hz 发送当前位姿
        '''
        if self.map.pose[0] is not None:  # 元组不是 None
            send_msg = String()
            send_msg.data = f"@0,{int(self.map.pose[0]*100)},{int(self.map.pose[1]*100)},{self.map.pose[2]:.2f}\r"
            self.uart1_SendMsg_pub(send_msg)

    # 新增：处理目标点的回调函数
    def target_callback(self, msg: PoseStamped):
        """
        处理接收到的目标点信息，将目标点信息通过串口发送。
        Args:
            msg (PoseStamped): 包含目标点位置和姿态信息的消息。
        """
        # 先处理指令相关逻辑
        if self.current_command == 'r01':  # 请求目标点
            self.target_RCP = (None, None)
            self.get_logger().info("重发目标点！")
            if msg:  # 确保 msg 存在
                send_msg = String()
                send_msg.data = f"@2,{msg.pose.position.x:.2f},{msg.pose.position.y:.2f}\r"
                self.serial_send_callback(send_msg)
            self.current_command = None
        elif self.current_command == 'r02':
            self.target_index += 1
            if self.target_index >= len(self.roads):
                self.target_index = 0
            self.target_RCP = self.roads[self.target_index]
            send_msg = String()
            send_msg.data = f"@2,{self.target_RCP[0]:.2f},{self.target_RCP[1]:.2f}\r"
            self.get_logger().info(f"切换下一个目标点{self.target_index}: {send_msg.data}")
            self.serial_send_callback(send_msg)
            self.current_command = None

    # 新增：云台重置函数
    def set_gimbal_angle(self):
        """调用set_gimbal服务设置云台角度"""
        # 调用前检查服务是否就绪
        if not self.gimbal_set_client.service_is_ready():
            self.get_logger().error("云台设置服务未就绪")
            return
        # 创建空请求并异步调用服务
        req = Empty.Request()
        future = self.gimbal_set_client.call_async(req)
        # 处理响应的回调函数
        def handle_set_response(future):
            try:
                future.result()  # 获取响应（空响应）
                self.get_logger().info("打靶角度设置成功")
            except Exception as e:
                self.get_logger().error(f"打靶角度设置成功: {e}")
        
        future.add_done_callback(handle_set_response)

    def reset_gimbal_angle(self):
        """调用reset_gimbal服务重置云台角度"""
        # 调用前检查服务是否就绪
        if not self.gimbal_reset_client.service_is_ready():
            self.get_logger().error("云台重置服务未就绪，无法执行重置操作")
            return
        # 创建空请求并异步调用服务
        req = Empty.Request()
        future = self.gimbal_reset_client.call_async(req)
        
        # 处理响应的回调函数
        def handle_reset_response(future):
            try:
                future.result()  # 获取响应（空响应）
                self.get_logger().info("云台角度已成功重置")
            except Exception as e:
                self.get_logger().error(f"云台重置失败: {e}")
        
        future.add_done_callback(handle_reset_response)
   
    def tracking_callback(self):
        pass
    def state_machine(self):
        '''
        @0,x,y,yaw: 当前位置
        @1,x,y,yaw: 目标点坐标

        @2,distance,angle: 杯子方位
        @3,digit: 数字
        @4:offset: 车身偏移量
        @5: 中靶


        'r11': 原点启动
        'r12': 在C点请求杯子方位

        'r21': 在杯子处请求数字结果
        'r22': 请求返回点B\D
        'r31': 返回B\D点请求修正偏差
        'r32': 停止修正
        'r41': 返回中靶结果
        

        '''
        if self.current_command == 'r11':
            # 放下摄像头
            # 记录杯子方位
            self.get_logger().info(f"---------- 任务一  ----------")
            # 启动记录 --- 仅作为单片机的启动指令
            send_msg = String()
            send_msg.data = f"@2,{self.cup[2]},{self.cup[3]}\r"
            self.uart1_SendMsg_pub(send_msg)
            # 1.1 放下摄像头
            self.reset_gimbal_angle()  # 调用云台重置函数
            self.get_logger().info(f"放下摄像头")
            # 1.2 记录杯子坐标
            self.get_logger().info(f"记录杯子方位")
            # 1.2.1 笛卡尔相对坐标
            self.map.target_RCP = (self.cup[0], self.cup[1], self.cup[3])   # (x,y,yaw)
            self.get_logger().info(f"杯子相对笛卡尔坐标: {self.map.target_RCP}")
            # 1.2.2 笛卡尔绝对坐标
            if self.map.pose[0] and self.map.target_RCP[0] is not None:
                self.map.target_ACP = (self.map.target_RCP[0] + self.map.pose[0],
                                    self.map.target_RCP[1] + self.map.pose[1],
                                    self.map.target_RCP[2] + self.map.pose[2]
                                    )
                self.get_logger().info(f"杯子绝对笛卡尔坐标: {self.map.target_ACP}")
            else:
                self.get_logger().info(f"自身坐标: {self.map.target_ACP}")
            # 1.2.3 极坐标
            self.map.target_polar = (self.cup[2], self.cup[3])      # (dis, angle)
            # 标识最终入口
            if self.map.target_polar[1] > 0:
                self.enter = -1  # B
            else:
                self.enter = 1   # D
            
            self.get_logger().info(f"杯子极坐标: {self.map.target_polar}")
            # 1.3 重置状态
            self.current_command = None

        elif self.current_command == 'r12':
            # 1.1 发送杯子方位
            self.get_logger().info(f"存储点位")
            # send_msg = String()
            # send_msg.data = f"@2,{self.cup[2]},{self.cup[3]}\r"
            # self.uart1_SendMsg_pub(send_msg)
            # self.get_logger().info(f"杯子方位: {send_msg.data}")
            # 1.2 存储地图 C,B,D 点
            self.map.point_C = (self.map.pose[0], self.map.pose[1], 0)
            self.map.point_B = (self.map.point_C[0]-0.22,self.map.point_C[1]+0.35, -90.00)
            self.map.point_D = (self.map.point_C[0]-0.22,self.map.point_C[1]-0.35, 90.00)
            self.get_logger().info(f"地图C点: {self.map.point_C}\n, 地图B点: {self.map.point_B},\n 地图D点: {self.map.point_D}")
            # 1.3 重置状态
            self.current_command = None

        elif self.current_command == 'r13':
            # 1.1 发送杯子方位
            self.get_logger().info(f"发送杯子方位")
            send_msg = String()
            send_msg.data = f"@2,{self.cup[2]},{self.cup[3]}\r"
            self.uart1_SendMsg_pub(send_msg)
            self.current_command = None
        
        elif self.current_command == 'r21':
            # 发送数字识别结果
            self.get_logger().info(f"---------- 任务二 ----------")
            # 2.1 发送数字识别结果
            if self.digit is not None:
                send_msg = String()
                send_msg.data = f"@3,{self.digit}\r"
                self.uart1_SendMsg_pub(send_msg)
                self.get_logger().info(f"数字识别结果:{self.digit}")
                self.digit = None
                self.current_command = None
            else:
                self.get_logger().info(f"等待数字识别......")
                self.current_command = None
            # 2.3 重置状态
            
        
        elif self.current_command == 'r22':
            # 发送返回节点
            if self.enter == -1: # 如果在C点左边，返回B点
                # 发送 B 点坐标
                send_msg = String()   # ros2 String类型
                send_msg.data = f"@1,{int(self.map.point_B[0]*100)},{int(self.map.point_B[1]*100)},{self.map.point_B[2]},B\r"
                self.uart1_SendMsg_pub(send_msg)
                self.get_logger().info(f"返回B点: {send_msg.data}")
            else: # 如果在C点右边，返回D点
                # 发送 D 点坐标
                send_msg = String()
                send_msg.data = f"@1,{int(self.map.point_D[0]*100)},{int(self.map.point_D[1]*100)},{self.map.point_D[2]},D\r"
                self.uart1_SendMsg_pub(send_msg)
                self.get_logger().info(f"返回D点: {send_msg.data}")
            # 2.3 重置状态
            self.current_command = None
            
        elif self.current_command == 'r31':
            # 发送修正偏差 【像素偏移量】
            # 4.1 发送修正偏移量
            self.get_logger().info(f"---------- 任务三 ----------")
            self.get_logger().info(f"修正偏差:self.enter_offset")
            if self.enter_offset is not None:
                send_msg = String()
                send_msg.data = f"@4,{self.enter_offset}\r"
                self.uart1_SendMsg_pub(send_msg)
            # 4.2 一直发送，不重置
            # self.current_command = None
        
        elif self.current_command == 'r32':
            # 停止发送修正偏差
            self.get_logger().info(f"停止发送修正偏差")
            # 3.3 重置状态
            self.current_command = None

        elif self.current_command == 'r41':
            # 发送打靶结果
            if not self.shotting_flag:
                self.get_logger().info(f"---------- 任务四 ----------")
                # 3.1设置摄像头
                self.set_gimbal_angle()
                # 3.2 发送打靶结果
                self.get_logger().info(f"打靶结果:{self.shotting_result}")
                self.shotting_flag = True
            if self.shotting_result:
                send_msg = String()
                send_msg.data = "@5\r"
                self.shotting_result = False
                self.uart1_SendMsg_pub(send_msg)
            # 3.3 重置状态
            # self.current_command = None




    # ================== uart 函数 ===================
    # ====== 发送 ======
    def uart1_sender_init(self, topic_stack=10):
        '''
            在其他话题的初始化中使用
            创建一个发布方 ---- 发布方功能：发送数据给串口 
            功能：发送数据到 uart_sender_topic
        '''
        self.pub_uart1_sender = self.create_publisher(String, 'uart1_sender_topic', topic_stack)
        self.get_logger().info(f'创建uart1_sender_topic发布方成功！')  
    def uart1_SendMsg_pub(self, data):
        # 创建String消息对象
        send_msg = String()
        
        # 处理输入数据，确保是字符串类型
        if isinstance(data, String):
            msg_data = data.data  # 从String消息中获取字符串
        else:
            msg_data = str(data)  # 将其他类型转换为字符串
        
        # 检查字符串是否以\r结尾
        if not msg_data.endswith('\r'):
            self.get_logger().info(f'消息没有以 \\r 结尾，添加回车符')
            msg_data += '\r'
        
        # 设置消息内容
        send_msg.data = msg_data
        
        # 发布消息
        self.pub_uart1_sender.publish(send_msg)
        # self.get_logger().info(f'发送数据: {msg_data} 到 uart1')

    # ====== 接收 ======
    def uart1_receiver_init(self, topic_stack=10):
        '''
            在非串口节点中使用
            订阅 uart1_receiver_topic 话题，接收相关数据进行处理
            【需搭配定时器进行使用】
        '''
        self.sub_uart1_receiver = self.create_subscription(String, "uart1_receiver_topic",self.uart1_RecvMsg_sub, topic_stack)
        # 有数据发布在 uart1_receiver_topic 话题，即调用uart1_RecvMsg_sub
        
    def uart1_RecvMsg_sub(self, msg):
        '''
            在非串口节点中使用
            订阅 uart1_receiver_topic 话题，接收相关数据进行处理
            
        '''
        self.get_logger().info(f"uart1接收到数据{msg}")
        self.current_command = msg.data.split('\r')[0]
        self.get_logger().info(f"指令为{self.current_command}")

    def destroy_node(self):
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    main_controller = MainController()
    main_controller.get_logger().info("主控制器节点已启动!")
    try:
        rclpy.spin(main_controller)
    except KeyboardInterrupt:
        pass
    finally:
        main_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()