"""
    语音播报节点 --- 发送数据到串口
    功能：
        1. 语音控制
            1.1 订阅 uart3_receiver_topic 话题，获取语音指令
            1.2 发布数据到 uart1_receiver_topic 话题，控制任务执行
        2. 播报
            2.1 订阅 uart1_receiver_topic 话题，获取任务指令
            2.2 发布数据到 uart3_sender_topic 话题内，根据任务指令发布不同的语音指令

    1. 语音控制
        1.1 订阅 uart3_receiver_topic 收到 'task1' 指令，发布 'r11' 到话题 uart1_receiver_topic
        1.2 订阅 uart3_receiver_topic 收到 'task2' 指令，发布 'r21' 到话题 uart1_receiver_topic
        1.3 订阅 uart3_receiver_topic 收到 'task3' 指令，发布 'r31' 到话题 uart1_receiver_topic
        1.4 订阅 uart3_receiver_topic 收到 'task4' 指令，发布 'r41' 到话题 uart1_receiver_topic
    2. 播报
        2.1 订阅 uart1_receiver_topic 收到 'r11'，发送 task1' 到话题 uart3_sender_topic       播报进入任务一，巡线去C点
        2.2 订阅 uart1_receiver_topic 收到 'r21'，发送 'task2' 到话题 uart3_sender_topic       播报进入任务二，寻找纸杯
        2.3 订阅 uart1_receiver_topic 收到 'r31'，发送 'task3' 到话题 uart3_sender_topic       播报进入任务三，识别数字
        2.4 订阅 uart1_receiver_topic 收到 'r41'，发送 'task4' 到话题 uart3_sender_topic       播报进入任务四，返回入口


        uart1_receiver_topic --[r11->task1]--> speaker_node --[发送task1]--> uart3_sender_topic
        uart1_receiver_topic <--[r21<-task2]-- speaker_node <--[接收task2]-- uart3_receiver_topic


"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SpeakerNode(Node):
    def __init__(self):
        super().__init__('speaker_node')
        
        # 创建四个话题接口：两个发布者和两个订阅者
        # 语音控制相关
        self.uart3_sub = self.create_subscription(
            String,
            'uart3_receiver_topic',
            self.uart3_callback,
            10
        )
        self.uart1_pub = self.create_publisher(
            String,
            'uart1_receiver_topic',
            10
        )
        
        # 播报相关
        self.uart1_sub = self.create_subscription(
            String,
            'uart1_receiver_topic',
            self.uart1_callback,
            10
        )
        self.uart3_pub = self.create_publisher(
            String,
            'uart3_sender_topic',
            10
        )
        
        # 语音控制映射表：串口指令 -> 任务指令
        self.voice_to_task_map = {
            'task1': 'r11',  # 任务一：巡线去C点
            'task2': 'r21',  # 任务二：寻找纸杯
            'task3': 'r31',  # 任务三：识别数字
            'task4': 'r41',  # 任务四：返回入口
        }
        
        # 播报映射表：任务指令 -> 语音播报指令
        self.task_to_speak_map = {
            'r11': 'task1',  # 播报进入任务一
            'r12': 'task2',  # 播报进入任务二
            'r21': 'task3',  # 播报进入任务三
            'r41': 'task4',  # 播报进入任务四
            # 'r41': 'task5',  # 播报进入任务五
        }
        
        self.get_logger().info('语音播报节点已启动')

    def uart3_callback(self, msg):
        """处理来自uart3_receiver_topic的语音控制指令"""
        voice_command = msg.data.strip().replace('\r', '').replace('\n', '')
        self.get_logger().info(f'收到语音控制指令: [{voice_command}]')
        
        # 检查指令是否在映射表中
        if voice_command in self.voice_to_task_map:
            task_command = self.voice_to_task_map[voice_command]
            
            # 发布任务指令到uart1_receiver_topic
            task_msg = String()
            task_msg.data = task_command
            self.uart1_pub.publish(task_msg)
            self.get_logger().info(f'发布任务指令: {task_command} 到 uart1_receiver_topic')
        else:
            self.get_logger().warn(f'未知语音控制指令: {voice_command}')

    def uart1_callback(self, msg):
        """处理来自uart1_receiver_topic的任务指令，触发语音播报"""
        task_command = msg.data.strip().replace('\r', '').replace('\n', '')
        self.get_logger().info(f'收到任务指令: [{task_command}]')
        
        # 检查指令是否在映射表中
        if task_command in self.task_to_speak_map:
            speak_command = self.task_to_speak_map[task_command]
            
            # 发布语音播报指令到uart3_sender_topic
            speak_msg = String()
            speak_msg.data = speak_command
            self.uart3_pub.publish(speak_msg)
            self.get_logger().info(f'发布语音播报指令: {speak_command} 到 uart3_sender_topic')
        else:
            # 处理其他可能的任务指令，例如状态反馈等
            self.get_logger().info(f'收到非播报任务指令: {task_command}')

def main(args=None):
    rclpy.init(args=args)
    node = SpeakerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()